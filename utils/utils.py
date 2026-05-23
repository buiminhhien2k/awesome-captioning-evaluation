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

    metric_registry = {
        "clip-score": lambda: ClipScoreMetric(metric_name=name, **kwargs),
        "pac-score": lambda: ClipScoreMetric(metric_name=name, **kwargs),
        "pac-score++": lambda: ClipScoreMetric(metric_name=name, **kwargs),

        "polos": lambda: PolosMetric(device=kwargs.get("device")),
        "standard": lambda: StandardMetric(),
        "bert-score": lambda: BertScoreBasic("en"),
        "bert-score++": lambda: BertScoreImproved("en"),
        "clip-image-score": lambda: ClipImageScore(kwargs.get("device")),
        "umic-score": lambda: UmicScore(),
        "blip2-score": lambda: Blip2ScoreMetric(),
        "mid-score": lambda: MIDScore(kwargs.get("device")),
    }

    if name not in metric_registry:
        raise ValueError(f"Unknown metric: {name}")

    return metric_registry[name]()

def save_metric_scores_jsonl(scores, dataset, file_json, asset_dir="asset"):
    dataset_dir = os.path.join(asset_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    file_stem = os.path.splitext(os.path.basename(file_json))[0]

    save_path = os.path.join(
        dataset_dir,
        f"{file_stem}_scores.jsonl"
    )

    existing_rows = {}

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_rows[row["metric_name"]] = row

    for metric_name, metric_result in scores.items():
        existing_rows[metric_name] = {
            "metric_name": metric_name,
            "scores": [
                round(score, 7)
                for score in metric_result["score_per_cap"]
            ],
        }

    with open(save_path, "w", encoding="utf-8") as f:
        for row in existing_rows.values():
            f.write(json.dumps(row) + "\n")