import time
from typing import Any, override

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from metrics.base_metric import BaseMetric

class SpiceCustomed(Spice):
    def __init__(self):
        super().__init__()
        """
        The goal of this class is just postprocess the output of Spice metric 
        so that its output format matches its brother and sister
        """

    @override
    def compute_score(self, gts, res):
        original_result = super().compute_score(gts, res)
        overall = original_result[0]
        score_per_cap = [per_cap['All']['f'] for per_cap in original_result[1]]
        return overall, score_per_cap

class StandardMetric(BaseMetric):

    @property
    def requires_references(self) -> bool:
        return True

    def setup(self):
        self.load_model()

    def load_model(self, **kwargs) -> None:
        self.STANDARD_SCORERS = [
            ("BLEU", Bleu(4)),
            ("ROUGE", Rouge()),
            ("METEOR", Meteor()),
            ("CIDER", Cider()),
            ("SPICE", SpiceCustomed()),
        ]

    def compute_score(
            self,
            gen_cs: list[str],
            gts_cs: list[list[str]],
            **kwargs,
    ) -> dict[str, dict[str, Any]]:
        raw_scores = self._compute_all_standard_metrics(
            refs=gts_cs,
            cands=gen_cs,
        )

        return self._format_scores(raw_scores)

    def _compute_all_standard_metrics(
            self,
            refs: list[list[str]],
            cands: list[str],
    ) -> dict[str, dict[str, Any]]:
        scores = {}

        for metric_name, scorer in self.STANDARD_SCORERS:
            start_time = time.perf_counter()

            overall, score_per_cap = pycoco_eval(
                scorer=scorer,
                refs=refs,
                cands=cands,
            )

            elapsed_seconds = time.perf_counter() - start_time

            scores[metric_name] = {
                "overall": overall,
                "score_per_cap": score_per_cap,
                "time": elapsed_seconds,
            }

        return scores

    def _format_scores(
            self,
            raw_scores: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        formatted_scores = {}

        for metric_name, score_data in raw_scores.items():
            if metric_name == "BLEU":
                formatted_scores.update(
                    self._extract_bleu_scores(score_data)
                )
            else:
                formatted_scores[metric_name] = score_data

        return formatted_scores

    def _extract_bleu_scores(
            self,
            bleu_scores: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return {
            "BLEU-1": {
                "overall": bleu_scores["overall"][0],
                "score_per_cap": bleu_scores["score_per_cap"][0],
                "time": bleu_scores["time"]
            },
            "BLEU-4": {
                "overall": bleu_scores["overall"][-1],
                "score_per_cap": bleu_scores["score_per_cap"][-1],
                "time": bleu_scores["time"]
            },
        }

def pycoco_eval(scorer, refs, cands):
    '''
        scorer is assumed to have a compute_score function.
        refs is a list of lists of strings
        cands is a list of predictions
    '''
    refs = {i: ref for i, ref in enumerate(refs)}
    cands = {i: [cand] for i, cand in enumerate(cands)}
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores

