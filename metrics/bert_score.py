import torch
import numpy as np
from .base_metric import BaseMetric
from collections import defaultdict

from bert_score import score
from bert_score.utils import (
    get_bert_embedding,
    get_tokenizer,
    get_model,
    lang2model,
    model2layers)


class BertScoreBasic(BaseMetric):

    def __init__(self, lang="en"):
        self.lang = lang

    def load_model(self):
        pass

    def setup(self, use_tfidf=False):
        self.use_tfidf = use_tfidf
        # if not using tfidf, all words/tokens will be treated with same weight
        # if use tfidf, the weight will be retrieved from reference captions

    def compute_score(self, ims_cs, gen_cs, gts_cs=None, gts=None, gen=None):
        """
            please refer to this `score` method from the original author
            https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py
            the length of the `gts` and
        :param ims_cs: list of string: list of image_path
            ims_cs[i] is a path caption of image i-th
        :param gen_cs: list of string: list of candidate captions
            gen_cs[i] is a candidate caption of image i-th
        :param gts_cs: list of list of string: list of list of reference caption
            gts_cs[i] is a list of reference caption of image i-th
        :param gts:
        :param gen:
        :return:
        """
        scores = dict()
        # for refs, cand in zip(gts_cs, gen_cs):
        p, r, f1 = score(
            cands=gen_cs,
            refs=gts_cs,
            lang=self.lang,
            idf=self.use_tfidf
        )

        scores[f"{self.__class__.__name__} F1-score"] = f1.mean()
        return scores

class BertScoreImproved(BaseMetric):

    def __init__(self,
                 lang="en",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 ):
        self.lang = lang
        self.device = device
        # device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type, num_layers):
        self.model = get_model(model_type, num_layers).to(self.device)
        self.tokenizer = get_tokenizer(model_type, True)

    def setup(self):
        model_type = lang2model[self.lang] # "roberta-large"
        num_layers = model2layers[model_type] # 17 for RoBERTa Large
        self.load_model(model_type, num_layers)

    def compute_score(self, ims_cs, gen_cs, gts_cs=None, gts=None, gen=None):
        """

        :param ims_cs: list of string: list of image_path
        :param gen_cs: list of string: list of candidate captions
            gen_cs[i] is a candidate caption of image i-th
        :param gts_cs: list of list of string: list of list of reference caption
            gts_cs[i] is a list of reference caption of image i-th
        :param gts:
        :param gen:
        :return:
        """
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[self.tokenizer.sep_token_id] = 0
        idf_dict[self.tokenizer.cls_token_id] = 0
        # TODO: allow idf weight based on reference captions like BertScoreBasic

        assert len(gts) == len(gen), "two `gts` and `gen` are not the same length"

        bert_scores = list()
        scores = dict()
        for refs, cand in zip(gts_cs[:10], gen_cs[:10]):
            ensembled_ref_matrix = self.get_ensemble_reference_word_vectors(
                refs, idf_dict, all_layers=False, default_threshold=0.83
            )
            vectored_cand_matrix = self.get_candidate_word_vectors(cand, idf_dict)
            p, r, f1 = self.compute_precision_recall_f1(ensembled_ref_matrix, vectored_cand_matrix)
            bert_scores.append(f1)

        scores[f"{self.__class__.__name__} F1-score"] = np.mean(bert_scores)

        return scores

    def get_ensemble_reference_word_vectors(self,
                                refs,
                                idf_dict,
                                default_threshold=0.83,
                                all_layers=False
                                ):
        '''
            refs: list of reference sentences
            idf_dict: dictionary of idf values for each token
            default_threshold=0.83, recommended threshold for RoBERTa (Large) produce 1024 dimension for each word
            all_layers=False: whether to return all layers or just the last layer

            return: emsembled_ref_matrix: (K', 1024) matrix of reference word vectors
            where K' is the number of unique tokens in all references (after removing similar tokens)
        '''
        embedding, mask, padded_idf = get_bert_embedding(
            refs, self.model,
            self.tokenizer, idf_dict,
            device=self.device, all_layers=all_layers)
        embeded_ref_norm  = embedding / (embedding * embedding).sum(axis=2, keepdims=True).sqrt()
        # token_vectors = (torch.unsqueeze(embeded_ref[1], 2) * embeded_ref_norm)  # (5, N, 1024)

        emsembled_ref_matrix = embeded_ref_norm[0][padded_idf[0].bool()] # initialize with the first reference
        # emsembled_ref_matrix: (K', 1024), where K' is the number of tokens in the first reference

        for i in range(1, embeded_ref_norm.shape[0]):
            # will update the K' in emsembled_ref_matrix to add more new tokens if there are any
            current_ref_matrix = embeded_ref_norm[i][padded_idf[i].bool()] # (K, 1024)
            # print(f"shape of reference {i}", current_ref_matrix.shape)
            confusion_mat = emsembled_ref_matrix @ current_ref_matrix.T

            assert torch.all(confusion_mat <= torch.tensor(1.0)) and torch.all(confusion_mat >= torch.tensor(-1.0))

            max_val_sim, _ = torch.max(confusion_mat, dim=0)

            assert current_ref_matrix.shape[0] == confusion_mat.shape[1]
            assert current_ref_matrix.shape[0] == max_val_sim.shape[0]

            selected_word_vectors = current_ref_matrix[max_val_sim < default_threshold] # max cosine similarity of each word in current_ref_matrix with all words in emsembled_ref_matrix)
            emsembled_ref_matrix = torch.cat((emsembled_ref_matrix, selected_word_vectors), dim=0)

        assert (emsembled_ref_matrix * emsembled_ref_matrix).sum(axis=1).sqrt().min().item() > 0.99
        return emsembled_ref_matrix

    def get_candidate_word_vectors(self,
        cand,
        idf_dict,
        all_layers=False
    ):
        '''
        cand: candidate sentence
        model: bert model
        tokenizer: bert tokenizer
        idf_dict: dictionary of idf values for each token
        device: device to run the model on
        all_layers: whether to return all layers or just the last layer

        return: cand_matrix: (K, 1024) matrix of candidate word vectors
        where K is the number of tokens in the candidate sentence
        '''
        embedding, mask, padd = get_bert_embedding(
            cand,
            self.model,
            self.tokenizer,
            idf_dict,
            device=self.device,
            all_layers=all_layers)
        # embeded_cand: (layers, 1, max_len, hidden_size) if all_layers is True
        # or (1, 1, max_len, hidden_size) if all_layers is False

        embeded_cand_norm  = embedding / (embedding * embedding).sum(axis=2, keepdims=True).sqrt()
        cand_matrix = embeded_cand_norm[padd.bool()] # (K, 1024)

        assert (cand_matrix * cand_matrix).sum(axis=1).sqrt().min().item() > 0.99
        return cand_matrix

    def compute_precision_recall_f1(self, ensembled_ref_matrix, cand_matrix):
        '''
        ensembled_ref_matrix: (K', 1024) matrix of reference word vectors
        cand_matrix: (K, 1024) matrix of candidate word vectors

        return: precision, recall, f1
        '''
        confusion_mat = cand_matrix @ ensembled_ref_matrix.T
        assert torch.all(confusion_mat <= torch.tensor(1.0)) and torch.all(confusion_mat >= torch.tensor(-1.0))

        max_val_sim_cand, _ = torch.max(confusion_mat, dim=1)
        max_val_sim_ref, _ = torch.max(confusion_mat, dim=0)

        precision = max_val_sim_cand.mean().item()
        recall = max_val_sim_ref.mean().item()
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1
