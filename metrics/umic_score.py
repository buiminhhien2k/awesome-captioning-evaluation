from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2 import model_zoo

from detectron2.structures import Boxes

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
        self.imageEmbedder = ImageFeatureEmbedder(self.device, self.rcnn_file)
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

        # self.rank_output = torch.nn.Linear(self.umicModel.config.hidden_size, 1).to(self.device)
        # self.rank_output.weight.data = umic_state['itm_output.weight'].data[1:,:].to(self.device, dtype=torch.float32)
        # self.rank_output.bias.data = umic_state['itm_output.bias'].data[1:].to(self.device, dtype=torch.float32)
        # self.rank_output.eval()

        # self.pooler = self.umicModel.pooler.eval()


    def compute_score(
            self,
            ims_cs,
            gen_cs,
            gts_cs,
            gts,
            gen
        ):
        """
        :param ims_cs: Required List<String>, list of path to the image
        :param gen_cs: Required List<String>, list candidate caption
        :param gts_cs: Nullable
        :param gts: Nullable
        :param gen: Nullable
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
            # outputs = self.umicModel(
            #     input_ids=cand_input_ids,
            #     attention_mask=joint_mask,
            #     position_ids=position_ids,
            #     img_feat=img_feat,
            #     img_pos_feat=img_box,
            #     gather_index=gather_ids,
            #     output_all_encoded_layers=False
            # )
            # pooled_output = self.pooler(outputs)
            # scores = self.rank_output(pooled_output)
            scores = self.umicModel(
                batch=batch,
                compute_loss=False
            )


            # scores = self.umicModel(
            #     batch=batch,
            #     compute_loss=False
            # )

            rank_scores += [scores.squeeze().detach().cpu().numpy()]
        # this step is refer to UMIC repository
        umic_score = [1/(1+math.exp(-rank_score)) for rank_score in rank_scores] # sigmoid

        return {"umic-score": {
            "overall": sum(umic_score) / len(umic_score),
            "score_per_cap": umic_score
        }
        }

    def read_image(self, image_path):
        image = Image.open(image_path)
        return np.array(image)

class ImageFeatureEmbedder:
    """
    Generate image region features + object boxes in UNITER format.

    For Faster R-CNN R101-C4:
        img_feat: (1, N, 2048)
        img_pos:  (1, N, 7)

    N is top_k, usually 36.
    """

    def __init__(
            self,
            device="cuda",
            file="faster_rcnn_R_101_C4_3x.yaml",
            top_k=36,
            score_thresh=0.2,
    ):
        self.device = device
        self.top_k = top_k

        self.cfg = get_cfg()
        self.cfg.merge_from_file(f"config/COCO-Detection/{file}")
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            f"COCO-Detection/{file}"
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.DEVICE = device

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )

        self.predictor = DefaultPredictor(self.cfg)
        self.model = self.predictor.model.eval()

    def _boxes_to_uniter_7d(self, boxes, img_h, img_w):
        """
        Convert boxes to normalised UNITER-style 7D position features.

        boxes: numpy array [N, 4], format [x1, y1, x2, y2]
        return: numpy array [N, 7]
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        box_w = np.maximum(x2 - x1, 0)
        box_h = np.maximum(y2 - y1, 0)
        area = box_w * box_h

        pos = np.stack(
            [
                x1 / img_w,
                y1 / img_h,
                x2 / img_w,
                y2 / img_h,
                box_w / img_w,
                box_h / img_h,
                area / (img_w * img_h),
                ],
            axis=1,
        )

        return pos.astype("float32")

    def embed_image(self, img):
        """
        :param img: numpy image, RGB, shape [H, W, 3]
        :return:
            img_feat: torch.FloatTensor [1, N, 2048]
            img_pos:  torch.FloatTensor [1, N, 7]
        """
        img_h, img_w = img.shape[:2]

        # Detectron2 expects BGR
        img_bgr = img[:, :, ::-1]

        transform = self.aug.get_transform(img_bgr)
        img_trans = transform.apply_image(img_bgr)

        img_tensor = torch.as_tensor(
            img_trans.astype("float32").transpose(2, 0, 1),
            device=self.device,
        )

        inputs = [
            {
                "image": img_tensor,
                "height": img_h,
                "width": img_w,
            }
        ]

        with torch.no_grad():
            images = self.model.preprocess_image(inputs)
            features = self.model.backbone(images.tensor)

            proposals, _ = self.model.proposal_generator(
                images, features, None
            )

            # Run ROI heads to obtain final detected instances.
            results, _ = self.model.roi_heads(
                images, features, proposals, None
            )

            instances = results[0]

            if len(instances) == 0:
                raise RuntimeError("No detected boxes found for this image.")

            boxes = instances.pred_boxes.tensor
            scores = instances.scores

            # Keep top-K final detections.
            k = min(self.top_k, boxes.shape[0])
            topk = torch.argsort(scores, descending=True)[:k]
            boxes = boxes[topk]

            # C4 / Res5ROIHeads path.
            if not (
                    hasattr(self.model.roi_heads, "pooler")
                    and hasattr(self.model.roi_heads, "res5")
            ):
                raise RuntimeError(
                    f"This implementation expects C4 Res5ROIHeads, "
                    f"but got {type(self.model.roi_heads)}"
                )

            final_boxes = [Boxes(boxes)]

            box_features = self.model.roi_heads.pooler(
                [features[f] for f in self.model.roi_heads.in_features],
                final_boxes,
            )

            # [N, 2048, H, W]
            box_features = self.model.roi_heads.res5(box_features)

            # [N, 2048]
            box_features = box_features.mean(dim=[2, 3])

            region_features = box_features.detach().cpu().numpy().astype("float32")
            final_boxes_np = boxes.detach().cpu().numpy().astype("float32")

        pos_7d = self._boxes_to_uniter_7d(final_boxes_np, img_h, img_w)

        img_feat = region_features[None, :, :]
        img_pos = pos_7d[None, :, :]

        assert img_feat.shape[2] == 2048, f"Expected 2048 image dim, got {img_feat.shape[2]}"
        assert img_pos.shape[2] == 7, f"Expected 7 box dim, got {img_pos.shape[2]}"
        assert img_feat.shape[1] == img_pos.shape[1], (
            f"N mismatch: img_feat has {img_feat.shape[1]}, "
            f"img_pos has {img_pos.shape[1]}"
        )

        return (
            torch.from_numpy(img_feat).to(self.device),
            torch.from_numpy(img_pos).to(self.device),
        )
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
