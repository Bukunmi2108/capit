"""Build data/vocab.json from the full Flickr8k train split."""

from __future__ import annotations

import json

from capit.config import config
from capit.data.vocab import Vocab


def train_captions(records: list[dict]) -> list[list[str]]:
    captions = [s["tokens"] for r in records if r["split"] == "train" for s in r["sentences"]]
    if not captions:
        splits = sorted({r["split"] for r in records})
        raise ValueError(f"no train records found; available splits: {splits}")
    return captions


def build_vocab() -> None:
    records = json.loads(config.karpathy_json.read_text())["images"]
    vocab = Vocab.build(train_captions(records), config.min_freq)
    config.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(config.vocab_path)
    print(f"vocab: {len(vocab)} words (min_freq={config.min_freq}) -> {config.vocab_path}")


if __name__ == "__main__":
    build_vocab()
