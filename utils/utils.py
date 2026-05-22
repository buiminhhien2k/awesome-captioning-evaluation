import torch
import json
import os

from metrics.clip_image_score import ClipImageScore
from metrics.clip_score import ClipScoreMetric
from metrics.mid_score import MIDScore
from metrics.polos import PolosMetric
from metrics.standard import StandardMetric
from metrics.bert_score import BertScoreBasic, BertScoreImproved
from metrics.umic_score import UmicScore
from metrics.blip2_score import Blip2ScoreMetric


def get_metric(name, **kwargs):
    name = name.lower()
    if name == "clip-score" or name == "pac-score" or name == "pac-score++":
        return ClipScoreMetric(metric_name=name, **kwargs)
    elif name == "polos":
        return PolosMetric(device=kwargs.get("device"))
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
    elif name == "mid-score":
        return MIDScore(kwargs.get("device"))
    else:
        raise ValueError(f"Unknown metric: {name}")


def save_metric_scores_jsonl(scores, dataset, file_json, asset_dir="asset"):
    os.makedirs(asset_dir, exist_ok=True)

    save_path = os.path.join(
        asset_dir,
        f"{dataset}_{os.path.splitext(file_json)[0]}_scores.jsonl"
    )

    # Load existing rows
    existing_rows = {}

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_rows[row["metric_name"]] = row

    # Replace / update rows
    for metric_name, metric_result in scores.items():

        existing_rows[metric_name] = {
            "metric_name": metric_name,
            "scores": [round(score,7) for score in metric_result["score_per_cap"]]
        }

    # Rewrite file
    with open(save_path, "w", encoding="utf-8") as f:
        for row in existing_rows.values():
            f.write(json.dumps(row) + "\n")