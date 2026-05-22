import json
from pathlib import Path
from typing import Any

from datasets.base_dataset import BaseDataset


class RawImageCaptionDataset(BaseDataset):
    def __init__(
            self,
            dataset_name: str,
            image_dir: str,
            caption_file: str,
    ):
        super().__init__(dataset_name)

        self.image_dir = Path(image_dir)
        self.caption_path = Path(caption_file)

    def _load_json(self, path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        data = self._load_json(self.caption_path)

        self.samples = []

        for item in data:
            image_id = item["image"]
            caption = item["caption"]

            self.samples.append({
                "dataset": self.dataset_name,
                "candidate_file": self.caption_path.name,
                "image_id": image_id,
                "image_path": str(self.image_dir / image_id),
                "candidate": caption,
                "references": None,
                "human_score": item.get("human-score"),
                "metadata": item.get("metadata", {}),
            })