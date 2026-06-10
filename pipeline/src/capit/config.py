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
    min_freq: int = 5
    resize: int = 256
    crop: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    encoded_size: int = 14
    encoder_dim: int = 2048
    decoder_dim: int = 512
    attention_dim: int = 512
    embed_dim: int = 512
    dropout: float = 0.5
    alpha_c: float = 1.0
    decoder_lr: float = 4e-4
    batch_size: int = 64
    max_epochs: int = 50
    num_workers: int = 2
    grad_clip: float = 5.0
    patience: int = 10
    beam_k: int = 3
    beam_alpha: float = 0.7
    max_decode_len: int = 50


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

    @property
    def vocab_path(self) -> Path:
        return self.data_root / "vocab.json"

    @property
    def ckpt_dir(self) -> Path:
        return self.data_root / "checkpoints"


config = Config()
