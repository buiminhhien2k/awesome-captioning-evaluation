import argparse
import torch
import numpy as np
import os

from  scipy.stats import kendalltau, spearmanr, pearsonr
from utils.utils import prepare_json, get_metric

ACCEPTED_METRIC_TYPES = [
    "clip-score", "pac-score", "pac-score++",
    "polos", "standard", "bert-score", "bert-score++",
    "clip-image-score",
    "blip2-score",
    "umic-score"
]

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'open_clip_ViT-L/14', 'ViT-L/14'])
    parser.add_argument('--compute_metric_type', type=str, nargs='+',
                        default=['clip-score', 'pac-score', 'pac-score++'])

    parser.add_argument('--dataset', type=str,
                        default='flickrExpert',
                        choices=['flickrExpert', 'flickrCrowdflower', 'polaris', 'composite']
                        )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = []
    json_dir = f'test_captions/{args.dataset}'
    image_dir = f'data/{args.dataset}'
    for file_json in os.listdir(json_dir):
        if not file_json.endswith('.json') or file_json == 'reference_captions.json':
            continue
        # if file_json != "flickrExpert-wo-human.json": continue
        files.append(file_json)

    metrics_list = list()
    for metric_name in args.compute_metric_type:
        metric_obj = get_metric(metric_name, device="cuda",
                                clip_model=args.clip_model)
        if metric_name != 'standard':
            metric_obj.setup()

        metrics_list.append(metric_obj)
    for file_json in files:
        print(f"***************Processing file: {file_json}")
        ims_cs, gen_cs, gts_cs, human_scores = prepare_json(file_json, json_dir, image_dir)
        has_human_score = all([hs != None for hs in human_scores])
        for metric in metrics_list:
            scores = metric.compute_score(
                ims_cs=ims_cs, gen_cs=gen_cs, gts_cs=gts_cs)

            for k, v in scores.items():
                display_result_string = '%s: %.4f ' % (k, v["overall"])
                if has_human_score:
                    kt_b, _ = kendalltau(human_scores, v["score_per_cap"], variant='b')
                    kt_c, _ = kendalltau(human_scores, v["score_per_cap"], variant='c')
                    rho_s, _ = spearmanr(human_scores, v["score_per_cap"])
                    rho_p, _ = pearsonr(human_scores, v["score_per_cap"])
                    display_result_string = ('%s: %.4f,\tkendall-tau b: %.4f,\tkendall-tau c: %.4f,\tspearman: %.4f,\tpearson: %.4f,'
                                             % (k, v["overall"], kt_b, kt_c, rho_s, rho_p))
                print(display_result_string)
