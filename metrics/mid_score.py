import tqdm
import math
from .base_metric import BaseMetric
from models import clip
import torch
from PIL import Image
import time

class MIDScore(BaseMetric):
    METRIC_NAME = "MID-Score"

    def __init__(self, device=None):
        self.device = device
        self.prefix = 'A photo depicts '
        self.batch_size = 128
        self.mid_scaler_min, self.mid_scaler_max = -550 , 0
        self.refmid_scaler_min, self.refmid_scaler_max = -250 , 50
    @property
    def requires_references(self) -> bool:
        return True

    def load_model(self, **kwargs):
        """
        Loads the SDXL pipeline and applies VRAM-specific optimizations.
        """

        self.model, self.processor = clip.load(
            'ViT-B/32', device=self.device, download_root='./checkpoints/'
        )


    @torch.no_grad()
    def _encode_texts_to_cache(self, texts, processed_cap_dict, processed_cap_set, desc):
        new_texts = [
            text for text in dict.fromkeys(texts)
            if text not in processed_cap_set
        ]

        for i in tqdm.tqdm(range(0, len(new_texts), self.batch_size), desc=desc):
            batch_texts = new_texts[i:i + self.batch_size]

            tokenised = clip.tokenize(
                [self.prefix + text for text in batch_texts],
                truncate=True
            ).to(self.device)

            batch_embeds = self.model.encode_text(tokenised)

            for text, embed in zip(batch_texts, batch_embeds):
                processed_cap_dict[text] = embed.unsqueeze(0)
                processed_cap_set.add(text)


    @torch.no_grad()
    def _encode_images_to_cache(self, img_paths, processed_img_dict, processed_img_set):
        new_img_paths = [
            img_path for img_path in dict.fromkeys(img_paths)
            if img_path not in processed_img_set
        ]

        for i in tqdm.tqdm(range(0, len(new_img_paths), self.batch_size), desc="Encoding images"):
            batch_img_paths = new_img_paths[i:i + self.batch_size]

            batch_images = []
            for img_path in batch_img_paths:
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.processor(image)
                batch_images.append(image_tensor)

            batch_images = torch.stack(batch_images, dim=0).to(self.device)
            batch_embeds = self.model.encode_image(batch_images)

            for img_path, embed in zip(batch_img_paths, batch_embeds):
                processed_img_dict[img_path] = embed.unsqueeze(0)
                processed_img_set.add(img_path)


    def _flatten_references(self, gts_cs):
        return [ref for refs_list in gts_cs for ref in refs_list]


    def _build_candidate_embeddings(self, gen_cs, processed_cap_dict):
        return [processed_cap_dict[cand] for cand in gen_cs]


    def _build_image_and_ref_embeddings(self, ims_cs, gts_cs, processed_img_dict, processed_cap_dict):
        embedded_image_list = []
        embedded_refs_agg_list = []
        embedded_refs_raw_list = []

        for img_path, refs_list in zip(ims_cs, gts_cs):
            embedded_image_list.append(processed_img_dict[img_path])

            ref_embeds = [processed_cap_dict[ref] for ref in refs_list]
            mean_ref_embed = torch.cat(ref_embeds, dim=0).mean(dim=0, keepdim=True)

            embedded_refs_agg_list.append(mean_ref_embed)
            embedded_refs_raw_list.append(ref_embeds)

        return embedded_image_list, embedded_refs_agg_list, embedded_refs_raw_list


    def _prepare_mid_tensors(
            self,
            embedded_image_list,
            embedded_refs_agg_list,
            embedded_refs_raw_list,
            embedded_cand_list):
        Y = torch.cat(embedded_image_list, dim=0).to(torch.float64)
        X = torch.cat(embedded_refs_agg_list, dim=0).to(torch.float64)
        X_hat = torch.cat(embedded_cand_list, dim=0).to(torch.float64)

        Z = torch.cat([X, Y], dim=1).to(torch.float64)
        Z_hat = torch.cat([X_hat, Y], dim=1).to(torch.float64)

        cosine_list = list()
        for embedded_refs, embedded_cand in zip(embedded_refs_raw_list, embedded_cand_list):
            refs_mat = torch.concat(embedded_refs, dim=0) # |R| x D, with R is embedded_refs, D = 512 for clip-B/32
            refs_mat_normed = torch.nn.functional.normalize(refs_mat, dim=1)

            embedded_cand_norm = torch.nn.functional.normalize(embedded_cand, dim=1) # 1 x D
            cosine_sim_by_R = refs_mat_normed @ embedded_cand_norm.t() # |R| x 1

            cosine_list.append(max(cosine_sim_by_R.max().item(), 0))
        cosine_sim_tensor = torch.Tensor(cosine_list).to(self.device) # N

        N, D = X_hat.shape

        assert X.shape == (N, D)
        assert Y.shape == (N, D)
        assert X_hat.shape == (N, D)
        assert Z.shape == (N, 2 * D)
        assert Z_hat.shape == (N, 2 * D)
        assert cosine_sim_tensor.shape[0] == N

        return X, Y, X_hat, Z, Z_hat, cosine_sim_tensor

    def _mahalanobis_diag_in_batches(self, diff, sigma_inv, batch_size):
        scores = []

        for i in range(0, diff.shape[0], batch_size):
            batch_diff = diff[i:i + batch_size]

            # Equivalent to:
            # ((diff @ sigma_inv @ diff.T).diag())
            batch_scores = (batch_diff @ sigma_inv * batch_diff).sum(dim=1)

            scores.append(batch_scores)

        return torch.cat(scores, dim=0)

    def _compute_mid_scores(self, X, X_hat, Z, Z_hat, limit=30000):
        Sigma_X = self.covariance_matrix(X)
        Sigma_Z = self.covariance_matrix(Z)

        eye_X = torch.eye(Sigma_X.shape[0], device=self.device, dtype=Sigma_X.dtype)
        eye_Z = torch.eye(Sigma_Z.shape[0], device=self.device, dtype=Sigma_Z.dtype)

        Sigma_X_inv = torch.linalg.inv(Sigma_X + 2e-2 * eye_X)
        Sigma_Z_inv = torch.linalg.inv(Sigma_Z + 2e-2 * eye_Z)

        diff_X = X_hat - X
        diff_Z = Z_hat - Z

        if X.shape[0] > limit:
            batch_size = max(1, limit // 2)

            KL_x = self._mahalanobis_diag_in_batches(
                diff=diff_X,
                sigma_inv=Sigma_X_inv,
                batch_size=batch_size
            )

            KL_z = self._mahalanobis_diag_in_batches(
                diff=diff_Z,
                sigma_inv=Sigma_Z_inv,
                batch_size=batch_size
            )
        else:
            KL_x = (diff_X @ Sigma_X_inv * diff_X).sum(dim=1)
            KL_z = (diff_Z @ Sigma_Z_inv * diff_Z).sum(dim=1)

        assert KL_x.shape == KL_z.shape

        mid_scores = KL_x - KL_z

        return mid_scores

    def compute_score(self, ims_cs, gen_cs, gts_cs, **kwargs):
        start_time = time.perf_counter()

        processed_cap_set, processed_img_set = set(), set()
        processed_cap_dict, processed_img_dict = {}, {}

        # 1. Encode candidates
        self._encode_texts_to_cache(
            texts=gen_cs,
            processed_cap_dict=processed_cap_dict,
            processed_cap_set=processed_cap_set,
            desc="Encoding candidate captions"
        )

        # 2. Encode references
        all_refs = self._flatten_references(gts_cs)

        self._encode_texts_to_cache(
            texts=all_refs,
            processed_cap_dict=processed_cap_dict,
            processed_cap_set=processed_cap_set,
            desc="Encoding references"
        )

        # 3. Encode images
        self._encode_images_to_cache(
            img_paths=ims_cs,
            processed_img_dict=processed_img_dict,
            processed_img_set=processed_img_set
        )

        # 4. Build ordered embedding lists
        embedded_cand_list = self._build_candidate_embeddings(
            gen_cs=gen_cs,
            processed_cap_dict=processed_cap_dict
        )

        (embedded_image_list,
         embedded_refs_agg_list,
         embedded_refs_raw_list) = self._build_image_and_ref_embeddings(
            ims_cs=ims_cs,
            gts_cs=gts_cs,
            processed_img_dict=processed_img_dict,
            processed_cap_dict=processed_cap_dict
        )

        # 5. Prepare tensors
        X, Y, X_hat, Z, Z_hat, cosine_by_R = self._prepare_mid_tensors(embedded_image_list=embedded_image_list,
                                                          embedded_refs_agg_list=embedded_refs_agg_list,
                                                          embedded_refs_raw_list=embedded_refs_raw_list,
                                                          embedded_cand_list=embedded_cand_list)

        # 6. Compute per-caption MID scores
        mid_scores = self._compute_mid_scores(
            X=X,
            X_hat=X_hat,
            Z=Z,
            Z_hat=Z_hat
        )
        mid_scores = (mid_scores - self.mid_scaler_min) / (self.mid_scaler_max - self.mid_scaler_min)
        mid_score_time = time.perf_counter() - start_time

        ref_mid = (mid_scores + 1e2*cosine_by_R)/2
        ref_mid = (ref_mid - self.refmid_scaler_min) / (self.refmid_scaler_max - self.refmid_scaler_min)
        refmid_score_time = time.perf_counter() - start_time

        mid_scores = mid_scores.detach().cpu().tolist()

        ref_mid = ref_mid.detach().cpu().tolist()
        return {
            self.METRIC_NAME: {
                "overall": sum(mid_scores) / len(mid_scores),
                "score_per_cap": mid_scores,
                "time": mid_score_time
            },
            f"Ref{self.METRIC_NAME}": {
                "overall": sum(ref_mid) / len(ref_mid),
                "score_per_cap": ref_mid,
                "time": refmid_score_time
            }
        }
    def setup(self,):
        self.load_model()

    def covariance_matrix(self, A, unbiased=False, eps=3e-3):
        # A shape: N x D
        A_centered = A - A.mean(dim=0, keepdim=True)

        denom = A.shape[0] - 1 if unbiased else A.shape[0]
        return (A_centered.mT @ A_centered) / denom + eps * torch.eye(A.shape[1], device=self.device)

    def sigmoid(self, z ):
        return 1/(1+math.exp(z))