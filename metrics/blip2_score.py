import time
from collections import OrderedDict
from typing import Any

import torch
import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from metrics.base_metric import BaseMetric
from models.blip2.model.blip2_image_text_matching import Blip2ITM
from models.blip2.processor.blip_processor import (
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
)


class Blip2ScoreMetric(BaseMetric):
    METRIC_NAME = "BLIP2-score"

    def __init__(self, batch_size: int = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.cache_limit = 5000

        self.model_cls = Blip2ITM()
        self.model = None
        self.vis_processor = None
        self.txt_processor = None

    @property
    def requires_references(self) -> bool:
        return False

    def setup(self) -> None:
        self.load_model()

    def load_model(self, **kwargs) -> None:
        self.model = (
            self.model_cls
            .from_pretrained(model_type="pretrain_vitL")
            .to(self.device)
        )
        self.model.eval()

        cfg = OmegaConf.load(
            self.model_cls.default_config_path("pretrain_vitL")
        )

        self.vis_processor, self.txt_processor = self._load_processors(
            cfg.preprocess
        )

    def compute_score(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        if self.model is None or self.vis_processor is None:
            raise RuntimeError(
                "BLIP2 model is not initialized. Call setup() first."
            )

        if len(ims_cs) != len(gen_cs):
            raise ValueError(
                "Length mismatch: `ims_cs` and `gen_cs` must have the same length."
            )

        start_time = time.perf_counter()

        score_per_cap = self._compute_blip2_scores(
            ims_cs=ims_cs,
            gen_cs=gen_cs,
        )

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": sum(score_per_cap) / len(score_per_cap),
                "score_per_cap": score_per_cap,
                "time": elapsed_seconds,
            }
        }

    def _compute_blip2_scores(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
    ) -> list[float]:
        scores = []
        cached_processed_images = OrderedDict()

        with torch.no_grad():
            for start_idx in tqdm.tqdm(
                    range(0, len(ims_cs), self.batch_size)
            ):
                end_idx = min(len(ims_cs), start_idx + self.batch_size)

                image_batch = ims_cs[start_idx:end_idx]
                caption_batch = gen_cs[start_idx:end_idx]

                processed_images = [
                    self._get_cached_processed_image(
                        img_path=image_path,
                        cache=cached_processed_images,
                    )
                    for image_path in image_batch
                ]

                image_tensor = torch.cat(processed_images)

                itc_score = self.model(
                    {
                        "image": image_tensor,
                        "text_input": caption_batch,
                    },
                    match_head="itc",
                )

                scores.extend(itc_score[:, 0].tolist())

        return scores

    def _get_cached_processed_image(
            self,
            img_path: str,
            cache: OrderedDict,
    ) -> torch.Tensor:
        if img_path in cache:
            processed_image = cache.pop(img_path)
            cache[img_path] = processed_image
            return processed_image

        image = Image.open(img_path).convert("RGB")

        processed_image = (
            self.vis_processor(image)
            .unsqueeze(0)
            .to(self.device)
            .half()
        )

        cache[img_path] = processed_image

        if len(cache) > self.cache_limit:
            _, old_tensor = cache.popitem(last=False)
            del old_tensor

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return processed_image

    def _load_processors(self, config: DictConfig | ListConfig):
        vis_cfg = config.get("vis_processor")
        txt_cfg = config.get("text_processor")

        vis_eval_cfg = vis_cfg.get("eval") if vis_cfg is not None else None
        txt_eval_cfg = txt_cfg.get("eval") if txt_cfg is not None else None

        vis_processor = self._build_processor_from_cfg(vis_eval_cfg)
        txt_processor = self._build_processor_from_cfg(txt_eval_cfg)

        return vis_processor, txt_processor

    def _build_processor_from_cfg(self, cfg: DictConfig | ListConfig):
        if cfg is None:
            raise ValueError("Processor config cannot be None.")

        if cfg.name == "blip_image_eval":
            return BlipImageEvalProcessor.from_config(cfg)

        if cfg.name == "blip_caption":
            return BlipCaptionProcessor.from_config(cfg)

        raise ValueError(f"Unknown processor type: {cfg.name}")