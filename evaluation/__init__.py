'''
Automatic generation evaluation metrics wrapper
The most useful function here is
get_all_metrics(refs, cands)
'''
from typing import override

from .pac_score import PACScore, RefPACScore
from .tokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'BLEU'),
                               (Meteor(), 'METEOR'),
                               (Rouge(), 'ROUGE'),
                               (Cider(), 'CIDER'),
                               (SpiceCustomed(), 'SPICE')
                               ]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        metrics.append({
            'overall': overall,
            'score_per_cap': per_cap,
        })
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics

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
        print(score_per_cap)
        return overall, score_per_cap