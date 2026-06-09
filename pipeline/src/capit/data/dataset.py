"""CaptionDataset: (image, caption) pairs and evaluation views over a Flickr8k root."""

from __future__ import annotations

import json
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from capit.config import Config, config
from capit.data.vocab import END, PAD, START, Vocab


def build_transform(cfg: Config = config):
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize),
            transforms.CenterCrop(cfg.crop),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ]
    )


class CaptionDataset(Dataset):
    def __init__(self, root: Path | str, split: str, vocab: Vocab, transform) -> None:
        self.root = Path(root)
        self.split = split
        self.vocab = vocab
        self.transform = transform

        records = json.loads((self.root / "dataset_flickr8k.json").read_text())["images"]
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            available = sorted({r["split"] for r in records})
            raise ValueError(f"split {split!r} matched 0 records under {self.root}; available: {available}")
        self.pairs = [(r, s) for r in self.records for s in r["sentences"]]
        self.max_len = max(len(s["tokens"]) + 2 for r in self.records for s in r["sentences"])

        ordered = sorted(self.records, key=lambda r: r["filename"])
        self.image_id = {r["filename"]: i for i, r in enumerate(ordered)}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rec, sent = self.pairs[i]
        image = self._load_image(rec["filename"])
        ids = [self.vocab.word2id[START], *self.vocab.encode(sent["tokens"]), self.vocab.word2id[END]]
        caption_len = len(ids)
        pad = self.max_len - caption_len
        if pad < 0:
            raise ValueError(f"caption len {caption_len} exceeds max_len {self.max_len} for {rec['filename']!r}")
        ids += [self.vocab.word2id[PAD]] * pad
        return image, torch.tensor(ids), caption_len

    def _load_image(self, filename: str) -> torch.Tensor:
        with Image.open(self.root / "Images" / filename) as img:
            img = ImageOps.exif_transpose(img) or img
            return self.transform(img.convert("RGB"))

    def references(self) -> dict[int, list[list[str]]]:
        return {
            self.image_id[r["filename"]]: [s["tokens"] for s in r["sentences"]] for r in self.records
        }

    def iter_images(self):
        for r in self.records:
            yield self.image_id[r["filename"]], self._load_image(r["filename"])


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    images = torch.stack([img for img, _, _ in batch])
    captions = torch.stack([ids for _, ids, _ in batch])
    lengths = torch.tensor([clen for _, _, clen in batch])
    return images, captions, lengths
