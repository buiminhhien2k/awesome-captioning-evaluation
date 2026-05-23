from .base_dataset import BaseDataset
from .standard_caption_dataset import StandardCaptionDataset
from .reference_free_dataset import ReferenceFreeDataset

DATASET_CLASSES = {
    "StandardCaptionDataset": StandardCaptionDataset,
    "ReferenceFreeDataset": ReferenceFreeDataset,
}


def get_dataset_class(dataset_class_name: str):
    if dataset_class_name not in DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset class: {dataset_class_name}"
        )

    return DATASET_CLASSES[dataset_class_name]


def discover_candidate_files(dataset_name: str, config: dict):
    dataset_cls = get_dataset_class(config["dataset_class"])

    return dataset_cls.discover_candidate_files(
        dataset_name=dataset_name,
        captions_root=config.get("captions_root", "test_captions"),
        excluded_files=tuple(
            config.get("excluded_files", ["reference_captions.json"])
        ),
    )


def build_dataset(
        dataset_name: str,
        candidate_file: str,
        config: dict,
) -> BaseDataset:
    dataset_cls = get_dataset_class(config["dataset_class"])

    return dataset_cls(
        dataset_name=dataset_name,
        candidate_file=candidate_file,
        captions_root=config.get("captions_root", "test_captions"),
        image_root=config.get("image_root", "data"),
    )