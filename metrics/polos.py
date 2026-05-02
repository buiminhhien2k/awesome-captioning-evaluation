import numpy as np
from .base_metric import BaseMetric
from models.polos.models import download_model, load_checkpoint
from PIL import Image


class PolosMetric(BaseMetric):
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.metric_name = "polos"

    def setup(self):
        self.load_model()
    def load_model(self, **kwargs):
        model_path = download_model("polos")
        self.model = load_checkpoint(model_path)

    def prepare_polos_dict(self, ims_cs, gen_cs, gts_cs):
        polos_dict = []
        for i, (im, gen, gts) in enumerate(zip(ims_cs, gen_cs, gts_cs)):
            curr = {
                'img': Image.open(im).convert("RGB"),
                'mt': gen,
                'refs': gts
            }
            polos_dict.append(curr)
        return polos_dict

    def compute_score(self, ims_cs, gen_cs, gts_cs=None, gts=None, gen=None):
        if self.model is None:
            raise RuntimeError(
                "Polos model not initialized. Call setup() first.")

        self.polos_dict = self.prepare_polos_dict(ims_cs, gen_cs, gts_cs)

        _, scores = self.model.predict(
            self.polos_dict, batch_size=10, cuda=(self.device == "cuda"))
        # print(_, scores)
        return {f"{self.metric_name}": {
            "overall":np.mean(scores),
            "score_per_cap": scores}
        }
