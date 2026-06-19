import time
from typing import Any

from bert_score import score

from .base_metric import BaseMetric


class BertScoreBasic(BaseMetric):
    METRIC_NAME = "BERTScore"

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.use_tfidf = True

    @property
    def requires_references(self) -> bool:
        return True

    def setup(self, use_tfidf: bool = True) -> None:
        self.use_tfidf = use_tfidf

    def load_model(self, **kwargs) -> None:
        pass

    def compute_score(
            self,
            gen_cs: list[str],
            gts_cs: list[list[str]],
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        start_time = time.perf_counter()

        _, _, f1 = score(
            cands=gen_cs,
            refs=gts_cs,
            lang=self.lang,
            idf=self.use_tfidf,
            rescale_with_baseline=True,
        )

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": float(f1.mean()),
                "score_per_cap": f1.tolist(),
                "time": elapsed_seconds,
            }
        }