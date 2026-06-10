"""Build data/vocab.json from the full Flickr8k train split."""

from __future__ import annotations

import json

from capit.config import config
from capit.data.records import train_captions
from capit.data.vocab import Vocab


def build_vocab() -> None:
    records = json.loads(config.karpathy_json.read_text())["images"]
    vocab = Vocab.build(train_captions(records), config.min_freq)
    config.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(config.vocab_path)
    print(f"vocab: {len(vocab)} words (min_freq={config.min_freq}) -> {config.vocab_path}")


if __name__ == "__main__":
    build_vocab()
