import time
import numpy as np
from PIL import Image

from .base_metric import BaseMetric
from models.polos.models import download_model, load_checkpoint


class PolosMetric(BaseMetric):
    METRIC_NAME = "POLOS"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @property
    def requires_references(self) -> bool:
        return True

    def setup(self) -> None:
        self.load_model()

    def load_model(self, **kwargs) -> None:
        model_path = download_model("polos")
        self.model = load_checkpoint(model_path)

    def compute_score(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
            gts_cs: list[list[str]],
            **kwargs,
    ):
        start_time = time.perf_counter()
        if self.model is None:
            raise RuntimeError(
                "POLOS model not initialized. Call setup() first."
            )

        polos_inputs = self._prepare_inputs(
            ims_cs=ims_cs,
            gen_cs=gen_cs,
            gts_cs=gts_cs,
        )

        _, scores = self.model.predict(
            polos_inputs,
            batch_size=10,
            cuda=(self.device == "cuda"),
        )

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": float(np.mean(scores)),
                "score_per_cap": scores,
                "time": elapsed_seconds,
            }
        }

    def _prepare_inputs(
            self,
            ims_cs: list[str],
            gen_cs: list[str],
            gts_cs: list[list[str]],
    ) -> list[dict]:
        return [
            {
                "img": Image.open(image_path).convert("RGB"),
                "mt": candidate_caption,
                "refs": references,
            }
            for image_path, candidate_caption, references in zip(
                ims_cs,
                gen_cs,
                gts_cs,
            )
        ]