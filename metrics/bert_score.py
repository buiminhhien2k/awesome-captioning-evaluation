import math
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import tqdm
from bert_score import score
from bert_score.utils import (
    get_bert_embedding,
    get_model,
    get_tokenizer,
    lang2model,
    model2layers,
)

from .base_metric import BaseMetric


class BertScoreBasic(BaseMetric):
    METRIC_NAME = "BERTScore"

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.use_tfidf = True

    @property
    def requires_references(self) -> bool:
        return True

    def setup(self, use_tfidf: bool = True) -> None:
        self.use_tfidf = use_tfidf

    def load_model(self, **kwargs) -> None:
        pass

    def compute_score(
            self,
            gen_cs: list[str],
            gts_cs: list[list[str]],
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        start_time = time.perf_counter()

        _, _, f1 = score(
            cands=gen_cs,
            refs=gts_cs,
            lang=self.lang,
            idf=self.use_tfidf,
            rescale_with_baseline=True,
        )

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": float(f1.mean()),
                "score_per_cap": f1.tolist(),
                "time": elapsed_seconds,
            }
        }


class BertScoreImproved(BaseMetric):
    METRIC_NAME = "BERTScore++"

    def __init__(
            self,
            lang: str = "en",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.lang = lang
        self.device = device
        self.model = None
        self.tokenizer = None

    @property
    def requires_references(self) -> bool:
        return True

    def setup(self) -> None:
        model_type = lang2model[self.lang]
        num_layers = model2layers[model_type]
        self.load_model(model_type, num_layers)

    def load_model(self, model_type: str, num_layers: int) -> None:
        self.model = get_model(model_type, num_layers).to(self.device)
        self.tokenizer = get_tokenizer(model_type, True)

    def compute_score(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
            gts_cs: list[list[str]],
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "BERTScore++ model not initialized. Call setup() first."
            )

        start_time = time.perf_counter()

        idf_dict = self.build_ref_idf_dict(ims_cs, gts_cs)
        bert_scores = []

        cached_ref_vectors = {}
        cached_cand_vectors = {}

        for img_path, refs, cand in tqdm.tqdm(
                zip(ims_cs, gts_cs, gen_cs),
                total=len(gts_cs),
        ):
            if img_path not in cached_ref_vectors:
                cached_ref_vectors[img_path] = (
                    self.get_ensemble_reference_word_vectors(
                        refs=refs,
                        real_idf_dict=idf_dict,
                    )
                )

            ensembled_ref_matrix, ensembled_ref_idf = cached_ref_vectors[img_path]

            if cand not in cached_cand_vectors:
                cached_cand_vectors[cand] = self.get_candidate_word_vectors([cand])

            cand_matrix = cached_cand_vectors[cand]

            _, _, f1 = self.compute_precision_recall_f1(
                ensembled_ref_matrix=ensembled_ref_matrix,
                cand_matrix=cand_matrix,
                ref_idf=ensembled_ref_idf,
            )

            bert_scores.append(f1)

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": float(np.mean(bert_scores)),
                "score_per_cap": bert_scores,
                "time": elapsed_seconds,
            }
        }

    def get_ensemble_reference_word_vectors(
            self,
            refs: list[str],
            real_idf_dict,
            default_threshold: float = 0.83,
            all_layers: bool = False,
    ):
        flat_idf_dict = defaultdict(lambda: 1.0)
        flat_idf_dict[self.tokenizer.sep_token_id] = 0.0
        flat_idf_dict[self.tokenizer.cls_token_id] = 0.0

        embedding, _, padded_flat_idf = get_bert_embedding(
            refs,
            self.model,
            self.tokenizer,
            idf_dict=flat_idf_dict,
            device=self.device,
            all_layers=all_layers,
        )

        padded_real_idf = self._build_padded_real_idf(
            refs=refs,
            real_idf_dict=real_idf_dict,
            max_len=padded_flat_idf.shape[1],
        )

        embedded_ref_norm = self._normalize_embedding(embedding)

        first_ref_mask = padded_flat_idf[0].bool()
        ensembled_ref_matrix = embedded_ref_norm[0][first_ref_mask]
        ensembled_ref_idf = padded_real_idf[0][first_ref_mask]

        for i in range(1, embedded_ref_norm.shape[0]):
            current_ref_mask = padded_flat_idf[i].bool()

            current_ref_matrix = embedded_ref_norm[i][current_ref_mask]
            current_ref_idf = padded_real_idf[i][current_ref_mask]

            if current_ref_matrix.shape[0] == 0:
                continue

            similarity_matrix = ensembled_ref_matrix @ current_ref_matrix.T
            max_similarity, _ = torch.max(similarity_matrix, dim=0)

            selected_mask = max_similarity < default_threshold

            ensembled_ref_matrix = torch.cat(
                (ensembled_ref_matrix, current_ref_matrix[selected_mask]),
                dim=0,
            )

            ensembled_ref_idf = torch.cat(
                (ensembled_ref_idf, current_ref_idf[selected_mask]),
                dim=0,
            )

        assert ensembled_ref_matrix.shape[0] == ensembled_ref_idf.shape[0]

        return ensembled_ref_matrix, ensembled_ref_idf

    def get_candidate_word_vectors(
            self,
            cand: list[str],
            all_layers: bool = False,
    ):
        flat_idf_dict = defaultdict(lambda: 1.0)
        flat_idf_dict[self.tokenizer.sep_token_id] = 0.0
        flat_idf_dict[self.tokenizer.cls_token_id] = 0.0

        embedding, _, padded_idf = get_bert_embedding(
            cand,
            self.model,
            self.tokenizer,
            flat_idf_dict,
            device=self.device,
            all_layers=all_layers,
        )

        embedded_cand_norm = self._normalize_embedding(embedding)
        cand_matrix = embedded_cand_norm[padded_idf.bool()]

        assert (cand_matrix * cand_matrix).sum(axis=1).sqrt().min().item() > 0.99

        return cand_matrix

    def compute_precision_recall_f1(
            self,
            ensembled_ref_matrix,
            cand_matrix,
            ref_idf,
    ):
        similarity_matrix = cand_matrix @ ensembled_ref_matrix.T

        max_sim_cand, matched_ref_idx = torch.max(similarity_matrix, dim=1)
        max_sim_ref, _ = torch.max(similarity_matrix, dim=0)

        ref_idf = ref_idf.to(similarity_matrix.device)

        cand_weights = ref_idf[matched_ref_idx]

        precision = (
                (max_sim_cand * cand_weights).sum()
                / cand_weights.sum().clamp_min(1e-8)
        )

        recall = (
                (max_sim_ref * ref_idf).sum()
                / ref_idf.sum().clamp_min(1e-8)
        )

        precision = precision.item()
        recall = recall.item()

        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

        return precision, recall, f1

    def build_ref_idf_dict(
            self,
            ims_cs: list[str],
            gts_cs: list[list[str]],
    ):
        doc_freq = Counter()
        num_docs = 0
        processed_images = set()

        for img_path, refs in zip(ims_cs, gts_cs):
            if img_path in processed_images:
                continue

            processed_images.add(img_path)
            num_docs += 1

            unique_token_ids = set()

            for ref in refs:
                token_ids = self.tokenizer.encode(ref, add_special_tokens=True)
                unique_token_ids.update(token_ids)

            doc_freq.update(unique_token_ids)

        idf_dict = defaultdict(lambda: math.log((num_docs + 1) / 1))

        for token_id, df in doc_freq.items():
            idf_dict[token_id] = math.log((num_docs + 1) / (df + 1))

        idf_dict[self.tokenizer.sep_token_id] = 0.0
        idf_dict[self.tokenizer.cls_token_id] = 0.0

        return idf_dict

    def _build_padded_real_idf(
            self,
            refs: list[str],
            real_idf_dict,
            max_len: int,
    ):
        padded_real_idf = []

        for ref in refs:
            token_ids = self.tokenizer.encode(ref, add_special_tokens=True)
            token_ids = token_ids[:max_len]

            real_idfs = [real_idf_dict[token_id] for token_id in token_ids]
            real_idfs += [0.0] * (max_len - len(real_idfs))

            padded_real_idf.append(real_idfs)

        return torch.tensor(
            padded_real_idf,
            dtype=torch.float,
            device=self.device,
        )

    def _normalize_embedding(self, embedding):
        return embedding / (
            (embedding * embedding).sum(axis=2, keepdims=True).sqrt()
        )