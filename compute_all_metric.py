import argparse
from typing import Any

import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

import utils.utils
from datasets import build_dataset, discover_candidate_files, BaseDataset
from metrics.base_metric import BaseMetric
from utils.config import load_dataset_config
from utils.utils import get_metric


ACCEPTED_METRIC_TYPES = [
    "clip-score",
    "pac-score",
    "pac-score++",
    "polos",
    "standard",
    "bert-score",
    "bert-score++",
    "clip-image-score",
    "blip2-score",
    "umic-score",
    "mid-score",
]

ACCEPTED_DATASETS = [
    "flickr8k",
    "polaris",
    "composite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute image caption evaluation metrics"
    )

    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-L/14"],
    )

    parser.add_argument(
        "--metrics_name",
        type=str,
        nargs="+",
        default=["clip-score", "pac-score", "pac-score++"],
        choices=ACCEPTED_METRIC_TYPES,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="flickrExpert",
        choices=ACCEPTED_DATASETS,
    )

    return parser.parse_args()


def load_metrics(
        metrics_name: list[str],
        device: str,
        clip_model: str,
) -> list[Any]:
    metrics = []

    for metric_name in metrics_name:
        metric = get_metric(
            metric_name,
            device=device,
            clip_model=clip_model,
        )
        metric.setup()

        metrics.append(metric)

    return metrics


def has_valid_human_scores(human_scores: list[float | None]) -> bool:
    return all(score is not None for score in human_scores)


def compute_correlations(
        human_scores: list[float],
        metric_scores: list[float],
) -> dict[str, float]:
    kt_b, _ = kendalltau(human_scores, metric_scores, variant="b")
    kt_c, _ = kendalltau(human_scores, metric_scores, variant="c")
    rho_s, _ = spearmanr(human_scores, metric_scores)
    rho_p, _ = pearsonr(human_scores, metric_scores)

    return {
        "kendall_tau_b": kt_b,
        "kendall_tau_c": kt_c,
        "spearman": rho_s,
        "pearson": rho_p,
    }


def print_metric_result(
        score_name: str,
        overall_score: float,
        time_sec: float,
        correlations: dict[str, float] | None = None,
) -> None:
    if correlations is None:
        print(
            f"{score_name}: {overall_score:.4f},\t"
            f"time[min]: {time_sec / 60:.2f}"
        )
        return

    print(
        f"{score_name}: {overall_score:.4f},\t"
        f"kendall-tau b: {correlations['kendall_tau_b']:.4f},\t"
        f"kendall-tau c: {correlations['kendall_tau_c']:.4f},\t"
        f"spearman: {correlations['spearman']:.4f},\t"
        f"pearson: {correlations['pearson']:.4f},\t"
        f"time [min]: {(time_sec / 60):.2f}"
    )


def evaluate_metric(
        metric: BaseMetric,
        dataset: BaseDataset,
        file_json: str,
        image_paths: list[str],
        candidate_captions: list[str],
        reference_captions: list[list[str] | None],
        human_scores: list[float | None],
) -> None:

    if metric.requires_references:
        dataset.require_references()

    scores = metric.compute_score(
        ims_cs=image_paths,
        gen_cs=candidate_captions,
        gts_cs=reference_captions,
    )

    utils.utils.save_metric_scores_jsonl(
        scores=scores,
        dataset=dataset.dataset_name,
        file_json=file_json,
    )

    can_compute_correlation = dataset.has_human_scores()

    for score_name, score_data in scores.items():
        correlations = None

        if can_compute_correlation:
            correlations = compute_correlations(
                human_scores=human_scores,
                metric_scores=score_data["score_per_cap"],
            )

        print_metric_result(
            score_name=score_name,
            overall_score=score_data["overall"],
            correlations=correlations,
            time_sec=score_data["time"]
        )


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_config = load_dataset_config(args.dataset)

    candidate_files = discover_candidate_files(
        dataset_name=args.dataset,
        config=dataset_config,
    )

    metrics = load_metrics(
        metrics_name=args.metrics_name,
        device=device,
        clip_model=args.clip_model,
    )

    for file_json in candidate_files:
        print(f"\n*************** Processing file: {file_json}")

        dataset = build_dataset(
            dataset_name=args.dataset,
            candidate_file=file_json,
            config=dataset_config,
        )
        dataset.load()

        image_paths, candidate_captions, reference_captions, human_scores = (
            dataset.as_columns()
        )

        for metric in metrics:
            print(f"\nEvaluating: {metric.__class__.__name__}")

            evaluate_metric(
                metric=metric,
                dataset=dataset,
                file_json=file_json,
                image_paths=image_paths,
                candidate_captions=candidate_captions,
                reference_captions=reference_captions,
                human_scores=human_scores,
            )


if __name__ == "__main__":
    main()