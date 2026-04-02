import torch
import json
import evaluation
import os

from metrics.clip_image_score import ClipImageScore
from metrics.clip_score import ClipScoreMetric
# from metrics.polos import PolosMetric
from metrics.standard import StandardMetric
from metrics.bert_score import BertScoreBasic, BertScoreImproved
from metrics.umic_score import UmicScore
from metrics.blip2_score import Blip2ScoreMetric


def collate_fn(batch):
    if isinstance(batch, tuple) and isinstance(batch[0], list):
        return batch
    elif isinstance(batch, list):
        transposed = list(zip(*batch))
        return [collate_fn(samples) for samples in transposed]
    return torch.utils.data.default_collate(batch)


def prepare_json(file_json, data_dir="test_captions"):
    with open(f'{data_dir}/reference_captions.json', 'r') as f:
        references = json.load(f)

    # check if file exist
    if os.path.isfile(f'{data_dir}/{file_json}'):
        with open(f'{data_dir}/{file_json}', 'r') as f:
            data = json.load(f)
    else:
        print(
            f"File {file_json} not found in test_captions/.")

    gen_tokenized = {}
    gts_tokenized = {}

    image_paths  : list[str] = list()
    cand_captions: list[str] = list()
    refs_captions: list[list[str]] = list()
    human_scores : list[float] = list()
    filejson_to_image_dir_mapper ={
        'flickrExpert-wo-human.json': 'flickr8k',
        'flickrExpert-w-human.json': 'flickr8k',
        'flickrCrowdflower.json': 'flickr8k',
    }
    for i, data in enumerate(data):  # k = name img, v=cand
        assert isinstance(data, dict)
        image_id: str = data["image-id"] # required field
        cand_caption: str = data["cand-caption"] # required field
        human_score: float|None = data.get("human-score", None) # optional field
        # human score can be null depending on the purpose of using this benchmark

        image_dir = filejson_to_image_dir_mapper[file_json]
        img_path = f'data/{image_dir}/{image_id}.jpg'

        refs_captions_i = references[image_id]
        gen_tokenized['%d' % (i)] = [cand_caption, ]
        gts_tokenized['%d' % (i)] = refs_captions_i

        image_paths.append(img_path)
        cand_captions.append(cand_caption)
        refs_captions.append(refs_captions_i)
        human_scores.append(human_score)

    gts_tokenized = evaluation.PTBTokenizer.tokenize(gts_tokenized)
    gen_tokenized = evaluation.PTBTokenizer.tokenize(gen_tokenized)


    return gts_tokenized, gen_tokenized, image_paths, cand_captions, refs_captions, human_scores


def get_metric(name, **kwargs):
    name = name.lower()
    if name == "clip-score" or name == "pac-score" or name == "pac-score++":
        return ClipScoreMetric(metric_name=name, **kwargs)
    # elif name == "polos":
    #     return PolosMetric(device=kwargs.get("device"))
    elif name == "standard":
        return StandardMetric()
    elif name == "bert-score":
        return BertScoreBasic("en")
    elif name == "bert-score++":
        return BertScoreImproved("en")
    elif name == "clip-image-score":
        return ClipImageScore(kwargs.get("device"))
    elif name == "umic-score":
        return UmicScore()
    elif name == "blip2-score":
        return Blip2ScoreMetric()
    else:
        raise ValueError(f"Unknown metric: {name}")
