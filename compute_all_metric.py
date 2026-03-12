import argparse
import torch
import numpy as np
import os

from  scipy.stats import kendalltau
from utils.utils import prepare_json, get_metric

ACCEPTED_METRIC_TYPES = [
    "clip-score", "pac-score", "pac-score++",
    "polos", "standard", "bert-score", "bert-score++",
    "clip-image-score"
]

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    parser.add_argument('--compute_metric_type', type=str, nargs='+',
                        default=['clip-score', 'pac-score', 'pac-score++'])

    parser.add_argument('--captions_dir', type=str, nargs='+',
                        default='flickr')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    filejson_to_image_dir_mapper ={
        'flickrExpert-wo-human.json': 'flickr8k',
        'flickrCrowdflower.json': 'flickr8k',
    }
    files = []
    json_dir = f'test_captions/{args.captions_dir}'
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
        gts, gen, ims_cs, gen_cs, gts_cs, human_scores = prepare_json(file_json, json_dir)
        has_human_score = all([hs != None for hs in human_scores])
        for metric in metrics_list:
            scores = metric.compute_score(
                ims_cs=ims_cs, gen_cs=gen_cs, gts_cs=gts_cs, gts=gts, gen=gen)

            for k, v in scores.items():
                display_result_string = '%s: %.4f ' % (k, v["overall"])
                if has_human_score:
                    kt_b, _ = kendalltau(human_scores, v["score_per_cap"], variant='b')
                    kt_c, _ = kendalltau(human_scores, v["score_per_cap"], variant='c')
                    display_result_string = '%s: %.4f,\tkendall-tau b: %.4f,\tkendall-tau c: %.4f' % (k, v["overall"], kt_b, kt_c)
                print(display_result_string)
