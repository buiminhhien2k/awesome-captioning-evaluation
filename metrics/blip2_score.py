
import torch
import numpy as np

from PIL import Image
from omegaconf import OmegaConf, ListConfig, DictConfig

from metrics.base_metric import BaseMetric
from models.blip2.model.blip2_image_text_matching import Blip2ITM
from models.blip2.processor.blip_processor import BlipImageEvalProcessor, BlipCaptionProcessor

from typing import Union


class Blip2ScoreMetric(BaseMetric):
    def __init__(self, batch_size=12):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cls = Blip2ITM()
        self.BATCH_SIZE = batch_size

    def setup(self):
        self.load_model()

    def load_model(self, **kwargs):
        self.model = self.model_cls.from_pretrained(model_type="pretrain").to(self.device)
        self.model.eval()

        cfg = OmegaConf.load(self.model_cls.default_config_path("pretrain"))
        self.vis_processor, self.txt_processor = self.load_process(cfg.preprocess)

    def _build_proc_from_cfg(self, cfg: Union[DictConfig, ListConfig]):
        assert cfg is not None
        assert cfg.name in ("blip_image_eval", "blip_caption")
        if cfg.name == "blip_image_eval":
            return BlipImageEvalProcessor.from_config(cfg)
        else :
            return BlipCaptionProcessor.from_config(cfg)

    def load_process(self, config: Union[DictConfig, ListConfig]):
        """
        Load preprocessor configs and construct preprocessors.

        If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

        Args:
            config (ListConfig, DictConfig): preprocessor configs.

        Returns:
            vis_processors (dict): preprocessors for visual inputs.
            txt_processors (dict): preprocessors for text inputs.

            Key is "train" or "eval" for processors used in training and evaluation respectively.
        """

        vis_proc_cfg = config.get("vis_processor")
        txt_proc_cfg = config.get("text_processor")
        if vis_proc_cfg is not None:
            vis_eval_cfg = vis_proc_cfg.get("eval")
        else:
            vis_eval_cfg = None
        vis_processor = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_eval_cfg = txt_proc_cfg.get("eval")
        else:
            txt_eval_cfg = None

        txt_processor = self._build_proc_from_cfg(txt_eval_cfg)

        return vis_processor, txt_processor


    def compute_score(self, ims_cs, gen_cs, **kwargs):
        """
        :param ims_cs: Required List<String>, list of path to the image
        :param gen_cs: Required List<String>, list candidate caption
        :return:
        """

        assert len(ims_cs) == len(gen_cs), "len of `ims_cs` must be equal to len of `gen_cs`"
        N = len(ims_cs)

        blip2_score = list()
        for i in range(0, N, self.BATCH_SIZE):
            l_idx, u_idx = i, min(N, i + self.BATCH_SIZE)
            ims_cs_batched = ims_cs[l_idx: u_idx]
            gen_cs_batched = gen_cs[l_idx: u_idx]

            img_processed_list = list()
            for img_path in ims_cs_batched:
                img_pil: Image.Image = Image.open(img_path)
                img: torch.Tensor = self.vis_processor(img_pil).unsqueeze(0).to(self.device).half()
                # size of img is (1, 3, 224, 224)
                img_processed_list.append(img)
            img_cs_batched = torch.concat(img_processed_list)
            # size of img is (BATCH_SIZE, 3, 224, 224)

            itc_score: torch.Tensor = self.model({"image": img_cs_batched, "text_input": gen_cs_batched}, match_head='itc')
            blip2_score += itc_score[:, 0].tolist()

        return {
            "Blip2-score":{
                "overall": sum(blip2_score) / N,
                "score_per_cap": blip2_score
            }
        }