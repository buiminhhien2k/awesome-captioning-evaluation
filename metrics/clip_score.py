import time
from typing import Any

import numpy as np
import torch

from .base_metric import BaseMetric
from evaluation import PACScore, RefPACScore
from models.clip import clip
from models.clip_lora import clip_lora
from utils.config import load_model_paths


class ClipScoreMetric(BaseMetric):
    SCORE_WEIGHTS = {
        "clip-score": 2.5,
        "pac-score": 2.0,
        "pac-score++": 2.5,
    }

    def __init__(
            self,
            device: str,
            clip_model: str = "ViT-B/32",
            metric_name: str = "clip-score",
    ):
        if metric_name not in self.SCORE_WEIGHTS:
            raise ValueError(
                f"Unknown metric name: {metric_name}"
            )

        self.device = device
        self.clip_model = clip_model
        self.metric_name = metric_name
        self.weight = self.SCORE_WEIGHTS[metric_name]

        self.model_paths = load_model_paths()
        self.model = None
        self.preprocess = None

    @property
    def requires_references(self) -> bool:
        return False

    def setup(self) -> None:
        self.load_model()

    def load_model(self) -> None:
        model, preprocess = self._load_backbone_model()

        model = model.to(self.device).float()

        if self.metric_name.startswith("pac-score"):
            self._load_pac_checkpoint(model)

        model.eval()

        self.model = model
        self.preprocess = preprocess

    def compute_score(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
            gts_cs: list[list[str]] | None = None,
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call setup() first."
            )

        scores = {}

        start_time = time.perf_counter()

        mean_score, clip_scores, candidate_feats, _ = PACScore(
            self.model,
            self.preprocess,
            ims_cs,
            gen_cs,
            self.device,
            w=self.weight,
        )

        base_elapsed = time.perf_counter() - start_time

        metric_name = self._metric_display_name()

        scores[metric_name] = {
            "overall": mean_score,
            "score_per_cap": clip_scores.tolist(),
            "time": base_elapsed,
        }

        if self._has_references(gts_cs):

            ref_scores = self._compute_reference_scores(
                gts_cs=gts_cs,
                clip_scores=clip_scores,
                candidate_feats=candidate_feats,
            )

            ref_elapsed = time.perf_counter() - start_time

            scores[f"Ref_{metric_name}"] = {
                "overall": float(np.mean(ref_scores)),
                "score_per_cap": ref_scores.tolist(),
                "time": ref_elapsed,
            }

        return scores

    def _load_backbone_model(self):
        if self.metric_name == "pac-score++":
            print(f"Loading PAC-S++ backbone: {self.clip_model}")

            return clip_lora.load(
                self.clip_model,
                device=self.device,
                lora=4,
                download_root="./checkpoints/",
            )

        print(f"Loading CLIP backbone: {self.clip_model}")

        return clip.load(
            self.clip_model,
            device=self.device,
            download_root="./checkpoints/",
        )

    def _load_pac_checkpoint(self, model) -> None:
        checkpoint_key = f"{self.metric_name}_{self.clip_model}"
        checkpoint_path = self.model_paths[checkpoint_key]

        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        model.load_state_dict(checkpoint["state_dict"])

    def _compute_reference_scores(
            self,
            gts_cs,
            clip_scores,
            candidate_feats,
    ):
        _, text_text_scores = RefPACScore(
            self.model,
            gts_cs,
            candidate_feats,
            self.device,
        )

        return (
                2
                * clip_scores
                * text_text_scores
                / (clip_scores + text_text_scores)
        )

    def _metric_display_name(self) -> str:
        return f"{self.metric_name.upper()} ({self.clip_model})"

    def _has_references(self, gts_cs) -> bool:
        return (
                gts_cs is not None
                and all(ref is not None for ref in gts_cs)
        )