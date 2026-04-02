from .base_metric import BaseMetric
from evaluation import get_all_metrics


class StandardMetric(BaseMetric):
    def __init__(self):
        pass  # No setup or device needed for standard metrics

    def compute_score(self, gts, gen, ims_cs=None, gen_cs=None, gts_cs=None):
        all_scores = {}
        all_scores_metrics = get_all_metrics(gts_cs, gen_cs)

        for k, v in all_scores_metrics.items():
            if k == 'BLEU':
                all_scores['BLEU-1'] = {"overall": v["overall"][0], "score_per_cap": v["score_per_cap"][0]}
                all_scores['BLEU-4'] = {"overall": v["overall"][-1], "score_per_cap": v["score_per_cap"][-1]}
            else:
                all_scores[k] = v
        return all_scores

    def load_model(self, **kwargs):
        pass
