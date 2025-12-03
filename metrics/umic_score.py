import torch
import numpy as np
from .base_metric import BaseMetric

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2 import model_zoo

from PIL import Image

class UmicScore(BaseMetric):
    """
    This class reproduce the UMIC score which can be applied for ANY new dataset.
    That is the major difference from this class to the [UMIC](https://github.com/hwanheelee1993/UMIC).
    The original work already pre-embedded images of the common datasets like: COMPOSITE, FLICKR, etc.
    Because this class serve a more generic purpose so it would take longer time to run since
    it requires detectron2 to detect the bounding boxes and its corresponding feature vectors
    in any images (please view the UNITER model on how to use detectron2)
    """

    def __init__(self):
        pass
    def setup(self):
        # this class heavily depend on detectron2, is used to embed the input images
        self.imageEmbedder = ImageFeatureEmbedder()

    def load_model(self):
        pass

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

        for img_path, cand_cap in zip(ims_cs, gen_cs):
            image = self.read_image(img_path)
        return {"umic-score": 0}

    def read_image(self, image_path):
        image = Image.open(image_path)
        return np.array(image)

class ImageFeatureEmbedder:
    """
    This class is used to generate the image features and object boxes from image,
    which result would be served as input for UNITER model (with UMIC weight)
    """
    def __init__(self):
        """
        For the constructor to run properly please find download the
        `faster_rcnn_R_101_FPN_3x.yaml` and `Base-RCNN-FPN.yaml` in the correct directory.
        You can find the original source of these 2 yaml files in detectron2 repository
        https://github.com/facebookresearch/detectron2/tree/main/configs
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file("config/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        pass

    def embed_image(self, image_array):
        """

        :param image_array: shape of (Height, Width, channel)
        :return: a tuple of (array shape of [1, N, 2048], array shape of (1, N, 7))
        """
        pass