from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image

import torch.nn as nn

import torchvision.transforms.functional as F

"""
    HOW TO INSTALL Detectron2? if you cannot install with the standard method like this
    [instruction](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    please review this
    [discussion](https://github.com/facebookresearch/detectron2/discussions/5200).
    this is how I installed Detectron2
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+18f6958pt2.8.0cu129
"""
import torch
import numpy as np

import tqdm

from .base_metric import BaseMetric
from transformers import BertTokenizer

from models.uniter.ce import UniterForCaptioningMetric
from models.uniter.model import UniterModel
from utils.config import load_model_paths

from PIL import Image

import math

class UmicScore(BaseMetric):
    """
    This class reproduce the UMIC score which can be applied for ANY new dataset.
    That is the major difference from this class to the [UMIC](https://github.com/hwanheelee1993/UMIC).
    The original work already pre-embedded images of the common datasets like: COMPOSITE, FLICKR, etc.
    Because this class serve a more generic purpose so it would take longer time to run since
    it requires Detectron2 to detect the bounding boxes and its corresponding feature vectors
    in any images (please view the UNITER model on how to use detectron2)
    """

    def __init__(self, rcnn_file="faster_rcnn_R_101_C4_3x.yaml"):
        """
        :param rcnn_file: name of yaml file to configure detectron2
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rcnn_file = rcnn_file
        self.IMAGE_DIM = 1024 if self.rcnn_file == "faster_rcnn_R_101_FPN_3x.yaml" else 2048

        
    def setup(self):
        # this class heavily depend on detectron2, is used to embed the input images
        self.load_model()

    def load_model(self):
        self.imageEmbedder = ImageFeatureEmbedder(
            "config/COCO-Detection/" + "faster_rcnn_R_101_C4_caffe.yaml" ,
            'checkpoints/faster_rcnn_from_caffe_attr_original.pkl', device=self.device)
        self.candidateTextEmbedder = CandidateCaptionEmbedder(self.device)

        # You need to have `umic.pt` file in checkpoints folder
        # `umic.pt` can be download from here https://archive.org/download/umic_data/umic.pt sourced in
        # the author of UMIC metric repository
        # https://github.com/hwanheelee1993/UMIC?tab=readme-ov-file#-2-download-the-pretrained-model-
        umic_state = torch.load(load_model_paths()["umic"])
        self.umicModel = UniterForCaptioningMetric.from_pretrained(
            config_file="config/uniter-config/uniter-base.json",
            state_dict=umic_state,
            img_dim=self.IMAGE_DIM
        )
        self.umicModel.init_output()
        self.umicModel.to(self.device).eval()


    def compute_score(
            self,
            ims_cs,
            gen_cs,
            **kwargs
        ):
        """
        :param ims_cs: Required List<String>, list of path to the image
        :param gen_cs: Required List<String>, list candidate caption

        :return: Float, the UMIC score
        """

        assert len(ims_cs) == len(gen_cs), "list of ims_cs and gen_cs are expected to be the same"

        rank_scores = list()

        for img_path, cand_cap in tqdm.tqdm(zip(ims_cs, gen_cs), total=len(ims_cs)):
            # TODO: This version is currently calculate UMIC score for 1-by-1 image, next task is to
            #  use dataloader to process data in batch.
            image = self.read_image(img_path)
            img_feat, img_box = self.imageEmbedder.embed_image(image)
            img_mask = torch.ones(1, img_feat.shape[1], dtype=torch.long).to(self.device)

            cand_input_ids, cand_input_masks = self.candidateTextEmbedder.tokenize(cand_cap)

            # size of joint_mask is: N + L + 2
            # plus 2 tokens because the CLS (id=101) and SEP (id=102)
            # L is the number of token of cand_cap, or number of tokens of the longest caption in a batch
            joint_mask = torch.cat([img_mask, cand_input_masks], dim=1).to(self.device)
            position_ids = torch.arange(cand_input_ids.shape[1], dtype=torch.long, device=self.device)
            gather_ids = torch\
                .arange(cand_input_ids.shape[1] + img_feat.shape[1], dtype=torch.long, device=self.device)\
                .unsqueeze(0)

            batch = {
                "input_ids": cand_input_ids,
                "position_ids": position_ids,
                "img_feat": img_feat,
                "img_pos_feat": img_box,
                "attn_masks": joint_mask,
                "gather_index": gather_ids,
            }

            scores = self.umicModel(
                batch=batch,
                compute_loss=False
            )

            rank_scores += [scores.squeeze().detach().cpu().numpy()]
        # this step is refer to UMIC repository
        umic_score = [1/(1+math.exp(-rank_score)) for rank_score in rank_scores] # sigmoid

        return {"umic-score": {
            "overall": sum(umic_score) / len(umic_score),
            "score_per_cap": umic_score
        }
        }
    @classmethod
    def read_image(self, image_path: str) -> torch.Tensor:
        """
        Reads an image from disk and converts it to a normalized PyTorch tensor.

        Returns:
            Tensor of shape (3, H, W) with values in range [0.0, 1.0]
        """
        # 1. Open the image and force it to 3-channel RGB
        image = Image.open(image_path).convert('RGB')

        # 2. to_tensor() automatically converts the PIL Image to a FloatTensor,
        # permutes dimensions to (C, H, W), and scales pixels down from [0, 255] to [0.0, 1.0].
        image_tensor = F.to_tensor(image)

        return image_tensor

class ImageFeatureEmbedder:
    """
    Decoupled Detectron2 extraction pipeline using BUTD Caffe-ported weights.
    Implements the official Box Refinement and dynamic NMS logic for exact parity.
    """
    def __init__(self, cfg_path: str, weights_path: str, max_proposals: int = 36, device: str = 'cuda'):
        self.device = device
        self.max_proposals = max_proposals

        # 1. Setup Configuration
        self.cfg = get_cfg()
        self.cfg.set_new_allowed(True)
        self.cfg.merge_from_file(cfg_path)
        self.cfg.set_new_allowed(False)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = str(self.device)

        # 2. Build Model Natively
        self.model = build_model(self.cfg)

        # Patch RPN Head to match BUTD 512-channel architecture ---
        num_anchors = self.model.proposal_generator.rpn_head.anchor_deltas.out_channels // 4

        self.model.proposal_generator.rpn_head.conv = nn.Conv2d(
            1024, 512, kernel_size=3, stride=1, padding=1
        )
        self.model.proposal_generator.rpn_head.objectness_logits = nn.Conv2d(
            512, num_anchors, kernel_size=1, stride=1
        )
        self.model.proposal_generator.rpn_head.anchor_deltas = nn.Conv2d(
            512, num_anchors * 4, kernel_size=1, stride=1
        )

        self.model.proposal_generator.rpn_head.to(self.device)
        # --------------------------------------------------------------------------

        # 3. Load the weights AFTER patching
        checkpointer = DetectionCheckpointer(self.model)

        checkpoint_dict = checkpointer._load_file(self.cfg.MODEL.WEIGHTS)

        state_dict = checkpoint_dict["model"]
        for key in list(state_dict.keys()):
            if "attr" in key or "cls_embedding" in key:
                del state_dict[key]

        checkpointer._load_model(checkpoint_dict)
        self.model.eval()

        # 4. Explicitly define the transform for robust resizing
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )

    @torch.no_grad()
    def embed_image(self, image_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_input: PyTorch Tensor of shape (3, H, W) with values [0.0, 1.0] or [0, 255].
        Returns:
            img_feat: Tensor of shape (1, N, 2048)
            img_feat_pos: Tensor of shape (1, N, 7)
        """
        _, raw_height, raw_width = image_input.shape

        # --- A. Preprocessing ---
        raw_image = image_input.cpu().numpy().transpose(1, 2, 0)
        if raw_image.max() <= 1.0:
            raw_image = (raw_image * 255.0).astype(np.uint8)

        image = self.aug.get_transform(raw_image).apply_image(raw_image)

        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1), device=self.device)

        inputs = [{"image": image_tensor, "height": raw_height, "width": raw_width}]
        images = self.model.preprocess_image(inputs)

        # --- B. Core Feature Extraction (Delegated to Helper) ---
        visual_feats, final_boxes = self._extract_region_features(
            images,
            resized_hw=image_tensor.shape[1:],
            raw_height=raw_height,
            raw_width=raw_width
        )

        # --- C. Spatial Feature Extraction ---
        loc_feats = self._compute_location_features(final_boxes, raw_width, raw_height)

        return visual_feats.unsqueeze(0), loc_feats.unsqueeze(0)

    def _extract_region_features(self, images, resized_hw, raw_height, raw_width):
        """
        Helper function to run the backbone, RPN, RoI heads, and NMS.
        Returns visual features and refined bounding boxes.
        """
        # Run Backbone & RPN
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        proposal_boxes = [x.proposal_boxes for x in proposals]

        # RoI Transform & Res5
        features_list = [features[f] for f in self.model.roi_heads.in_features]
        box_features = self.model.roi_heads._shared_roi_transform(features_list, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])

        # Box Refinement
        predictions = self.model.roi_heads.box_predictor(feature_pooled)
        boxes = self.model.roi_heads.box_predictor.predict_boxes(predictions, proposals)[0]
        probs = self.model.roi_heads.box_predictor.predict_probs(predictions, proposals)[0]

        # Single-Pass NMS
        instances, ids = fast_rcnn_inference_single_image(
            boxes, probs, resized_hw,
            score_thresh=0.2, nms_thresh=0.5, topk_per_image=self.max_proposals
        )

        # Postprocess & Feature Selection
        instances = detector_postprocess(instances, raw_height, raw_width)
        visual_feats = feature_pooled[ids].detach()
        final_boxes = instances.pred_boxes.tensor

        return visual_feats, final_boxes

    def _compute_location_features(self, boxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
        """Computes the 7D normalized spatial vectors from the refined boxes."""
        if boxes.shape[0] == 0:
            return torch.zeros((1, 7), device=self.device)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        box_width, box_height = x2 - x1, y2 - y1

        loc_feats = torch.stack([
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
            box_width / image_width,
            box_height / image_height,
            (box_width * box_height) / (image_width * image_height)
        ], dim=1)

        return loc_feats

class CandidateCaptionEmbedder:
    def __init__(self, device="cuda"):
        """
        This class is a rework from UMIC repository, please check their original work in
        https://github.com/hwanheelee1993/UMIC/blob/master/make_txt_db.py
        :param device:
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def tokenize(self, cand_caption):
        """
        This function is basically just to "map" each token/word into a single number.
        The original work of UMIC author was to do it manually which does not include the [CLS]-101
        and [SEP]-102 token. Original version is
        https://github.com/hwanheelee1993/UMIC/blob/9d897ee575d754dada84e00da426bbceabffc450/make_txt_db.py#L18
        The only difference is my work has CLS at beginning and SEP at the end of each sequence
        :param cand_caption:
        :return:
        """

        tokens = self.tokenizer(
            cand_caption,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # input_ids = []
        # for word in cand_caption.split():
        #     ws = self.tokenizer.tokenize(word)
        #     if not ws:
        #         # some special char
        #         continue
        #     input_ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        # input_ids = torch.from_numpy(np.array(input_ids)).unsqueeze(0)
        # mask = torch.ones(input_ids.shape, dtype=torch.long)

        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)
        # return input_ids.to(self.device), mask.to(self.device)
