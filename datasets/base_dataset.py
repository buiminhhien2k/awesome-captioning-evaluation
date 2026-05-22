from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseDataset(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.samples: list[dict[str, Any]] = []

    @abstractmethod
    def load(self) -> None:
        pass

    @staticmethod
    def discover_candidate_files(
            dataset_name: str,
            captions_root: str = "test_captions",
            excluded_files: tuple[str, ...] = ("reference_captions.json",),
    ) -> list[str]:
        data_dir = Path(captions_root) / dataset_name

        if not data_dir.is_dir():
            raise FileNotFoundError(f"Dataset caption directory not found: {data_dir}")

        return sorted(
            path.name
            for path in data_dir.iterdir()
            if path.suffix == ".json" and path.name not in excluded_files
        )

    def has_references(self) -> bool:
        return any(sample.get("references") for sample in self.samples)

    def has_human_scores(self) -> bool:
        return any(sample.get("human_score") is not None for sample in self.samples)

    def require_references(self) -> None:
        if not self.has_references():
            raise ValueError(
                f"Dataset {self.dataset_name} does not contain references."
            )

    def require_human_scores(self) -> None:
        if not self.has_human_scores():
            raise ValueError(
                f"Dataset {self.dataset_name} does not contain human scores."
            )

    def as_columns(self):
        image_paths = [s["image_path"] for s in self.samples]
        candidates = [s["candidate"] for s in self.samples]
        references = [s.get("references") for s in self.samples]
        human_scores = [s.get("human_score") for s in self.samples]

        return image_paths, candidates, references, human_scores

    def get_batches(self, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]

    def __len__(self) -> int:
        return len(self.samples)