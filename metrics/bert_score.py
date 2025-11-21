
import torch
import numpy as np
from .base_metric import BaseMetric

from bert_score import score

class BertScoreBasic(BaseMetric):

    def __init__(self, lang="en"):
        self.lang = lang

    def setup(self, use_tfidf=False):
        self.use_tfidf = use_tfidf
        # if not using tfidf, all words/tokens will be treated with same weight
        # if use tfidf, the weight will be retrieved from reference captions

    def compute_score(self, ims_cs, gen_cs, gts_cs=None, gts=None, gen=None):
        """
            please refer to this `score` method from the original author
            https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py
            the length of the `gts` and
        :param ims_cs: List<String>, each string is a path to the image
        :param gen_cs: List<List<String>>, tokenized each candidate captions
        :param gts_cs: tokenized each reference captions
        :param gts: List<String>/List<List<String>>, list ground truth (reference) captions
        :param gen: List<String>, list candidate captions
        :return:
        """
        P_bert, R_bert, F_bert = score(
            cands=gen,
            refs=gts,
            lang=self.lang,
            idf=self.use_tfidf
        )
        return np.mean(F_bert)

class BertScoreImproved(BaseMetric):

    def __init__(self, lang="en"):
        self.lang = lang

    def setup(self):
        pass # doing nothing at the moment

    def compute_score(self, ims_cs, gen_cs, gts_cs=None, gts=None, gen=None):
        pass
