import tqdm
import math
from evaluation.pac_score.pac_score import extract_all_captions
from .base_metric import BaseMetric
from models import clip
import torch
from PIL import Image

class MIDScore(BaseMetric):
    def __init__(self, device=None):
        self.device = device
        self.prefix = 'A photo depicts '
        # self.negative_prompt = """(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, (NSFW:1.25)"""
        # self.width, self.height = 1024, 1024
        # self.pipe = None  # Initialize pipeline variable

    def load_model(self, **kwargs):
        """
        Loads the SDXL pipeline and applies VRAM-specific optimizations.
        """

        self.model, self.processor = clip.load(
            'ViT-B/32', device=self.device, download_root='./checkpoints/'
        )


    def compute_score(self, ims_cs, gen_cs, gts_cs, **kwargs):
        """
        :param ims_cs: list of image paths (not currently used in generation loop)
        :param gen_cs: list of candidate captions
        :return: dictionary with score
        """
        # Safety check: ensure model is loaded before computing

        embedded_image_list = list()
        embedded_cand_list = list()
        embedded_refs_list = list()

        processed_cap_set, processed_img_set = set(), set()
        processed_cap_dict, processed_img_dict = dict(), dict()

        for cand in tqdm.tqdm(gen_cs, total=len(gen_cs)):
            if cand not in processed_cap_set:
                tokenised_cand = clip.tokenize(self.prefix + cand, truncate=True)
                processed_cap_dict[cand] = self.model.encode_text(tokenised_cand.to(self.device))
                processed_cap_set.add(cand)
            embedded_cand_list.append(processed_cap_dict[cand])

        for img_path, refs_list in tqdm.tqdm(zip(ims_cs, gts_cs), total=len(ims_cs)):
            if img_path not in processed_img_set:
                original_image = Image.open(img_path)
                original_batch_tensor = self.processor(original_image).unsqueeze(0).to(self.device)
                processed_img_dict[img_path] = self.model.encode_image(original_batch_tensor)
                processed_img_set.add(img_path)
            embedded_image_list.append(processed_img_dict[img_path])

            list_embedded_ref_cap = list()
            for ref in refs_list:
                if ref not in processed_cap_set:
                    tokenised_ref = clip.tokenize(self.prefix + ref, truncate=True)
                    processed_cap_dict[ref] = self.model.encode_text(tokenised_ref.to(self.device))
                    processed_cap_set.add(ref)
                list_embedded_ref_cap.append(processed_cap_dict[ref])
            mean_embedded_ref = torch.mean(torch.concat(list_embedded_ref_cap), dim=0)
            embedded_refs_list.append(mean_embedded_ref.unsqueeze(dim=0))

        # embedded_cand_list, embedded_refs_list, embedded_image_list shape: N x D

        # shapes: N x D
        Y = torch.cat(embedded_image_list, dim=0).to(torch.float64)
        X = torch.cat(embedded_refs_list, dim=0).to(torch.float64)
        X_hat = torch.cat(embedded_cand_list, dim=0).to(torch.float64)

        Z = torch.cat([X, Y], dim=1).to(torch.float64)
        Z_hat = torch.cat([X_hat, Y], dim=1).to(torch.float64)

        Sigma_Y = self.covariance_matrix(Y)
        Sigma_X = self.covariance_matrix(X)
        Sigma_X_hat = self.covariance_matrix(X_hat)
        Sigma_Z = self.covariance_matrix(Z)
        Sigma_Z_hat = self.covariance_matrix(Z_hat)

        mu_Z = torch.mean(Z, dim=0).to(torch.float64)
        mu_Z_hat = torch.mean(Z_hat, dim=0)
        mu_X = torch.mean(X, dim=0).to(torch.float64)

        det_Sigma_X = torch.linalg.slogdet(Sigma_X)[1]
        det_Sigma_X_hat = torch.linalg.slogdet(Sigma_X_hat)[1]
        det_Sigma_Z = torch.linalg.slogdet(Sigma_Z)[1]
        det_Sigma_Z_hat = torch.linalg.slogdet(Sigma_Z_hat)[1]

        I_yx_hat = 1/2 * (
            det_Sigma_X_hat + torch.linalg.slogdet(Sigma_Y)[1] - det_Sigma_Z_hat
        )
        Sigma_X_inv = torch.linalg.inv(Sigma_X + 5e-2 * torch.eye(Sigma_X.shape[0], device=self.device))
        Sigma_Z_inv = torch.linalg.inv(Sigma_Z + 5e-2 * torch.eye(Sigma_Z.shape[0], device=self.device))

        mid_scores = list()

        KL_x_1 = torch.trace(Sigma_X_inv @ (Sigma_X_hat - Sigma_X))
        KL_x_2 = det_Sigma_X - det_Sigma_X_hat
        KL_z_1 = torch.trace(Sigma_Z_inv @ (Sigma_Z_hat - Sigma_Z))
        KL_z_2 = det_Sigma_Z - det_Sigma_Z_hat
        for i in tqdm.tqdm(range(len(ims_cs))):
            x_hat_i, z_hat_i = X_hat[i, :].unsqueeze(0), Z_hat[i, :].unsqueeze(0) # 1 x D
            x_i, z_i = X[i, :].unsqueeze(0), Z[i, :].unsqueeze(0)
            # print(x_hat_i.shape)
            # print(mu_X.shape)
            D_KL_x_i = 1/2 * (
                KL_x_1 +
                (x_hat_i - x_i) @ Sigma_X_inv @ (x_hat_i - x_i).mT +
                KL_x_2
            )
            D_KL_z_i = 1/2 * (
                KL_z_1 +
                (z_hat_i - z_i) @ Sigma_Z_inv @ (z_hat_i - z_i).mT +
                KL_z_2
            )
            score_i = (I_yx_hat + D_KL_x_i - D_KL_z_i).item()

            mid_scores.append(score_i)
        # mid_scores = [-(score_i-min(mid_scores)) / max(mid_scores) for score_i in mid_scores]
        return {"mid-score": {
            "overall": sum(mid_scores) / len(ims_cs),
            "score_per_cap": mid_scores
        }}

    def setup(self,):
        self.load_model()

    def covariance_matrix(self, A, unbiased=False, eps=1e-2):
        # A shape: N x D
        A_centered = A - A.mean(dim=0, keepdim=True)

        denom = A.shape[0] - 1 if unbiased else A.shape[0]
        return (A_centered.mT @ A_centered) / denom + eps * torch.eye(A.shape[1], device=self.device)

    def sigmoid(self, z ):
        return 1/(1+math.exp(z))