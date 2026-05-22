import json
from pathlib import Path
from typing import Any

from datasets.base_dataset import BaseDataset


class StandardCaptionDataset(BaseDataset):
    def __init__(
            self,
            dataset_name: str,
            candidate_file: str,
            captions_root: str = "test_captions",
            image_root: str = "data",
            reference_file: str = "reference_captions.json",
    ):
        super().__init__(dataset_name)

        self.data_dir = Path(captions_root) / dataset_name
        self.image_dir = Path(image_root) / dataset_name

        self.candidate_file = candidate_file
        self.candidate_path = self.data_dir / candidate_file
        self.reference_path = self.data_dir / reference_file

    def _load_json(self, path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        candidates = self._load_json(self.candidate_path)
        references = self._load_json(self.reference_path)

        self.samples = []

        for item in candidates:
            image_id = item["image-id"]

            self.samples.append({
                "dataset": self.dataset_name,
                "candidate_file": self.candidate_file,
                "image_id": image_id,
                "image_path": str(self.image_dir / image_id),
                "candidate": item["cand-caption"],
                "references": references.get(image_id),
                "human_score": item.get("human-score"),
                "metadata": {},
            })