"""
Single source of truth for every hyperparameter in the capit project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seed: int = 13
    data_root: Path = _REPO_ROOT / "data"
    subsample_train: int = 50
    subsample_val: int = 10
    subsample_test: int = 10

    @property
    def flickr8k_dir(self) -> Path:
        return self.data_root / "flickr8k"

    @property
    def images_dir(self) -> Path:
        return self.flickr8k_dir / "Images"

    @property
    def captions_txt(self) -> Path:
        return self.flickr8k_dir / "captions.txt"

    @property
    def karpathy_dir(self) -> Path:
        return self.data_root / "karpathy"

    @property
    def karpathy_json(self) -> Path:
        return self.karpathy_dir / "dataset_flickr8k.json"

    @property
    def subsample_counts(self) -> dict[str, int]:
        return {
            "train": self.subsample_train,
            "val": self.subsample_val,
            "test": self.subsample_test,
        }

    @property
    def subsample_root(self) -> Path:
        return self.data_root / "dev_subsample"

    @property
    def subsample_images_dir(self) -> Path:
        return self.subsample_root / "Images"

    @property
    def subsample_json(self) -> Path:
        return self.subsample_root / "dataset_flickr8k.json"


config = Config()
