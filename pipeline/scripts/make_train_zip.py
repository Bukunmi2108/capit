"""Assemble the full dataset + vocab into one zip for Colab (upload to Drive).
The zip unpacks to a single CaptionDataset root: Images/ + dataset_flickr8k.json + vocab.json.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from capit.config import config


def make_train_zip(out_path: Path | None = None) -> Path:
    out = Path(out_path) if out_path else config.data_root / "flickr8k_colab.zip"
    for required in (config.images_dir, config.karpathy_json, config.vocab_path):
        if not required.exists():
            raise FileNotFoundError(f"{required} missing — run the Stage 0.2 download and build_vocab first")

    jpgs = sorted(config.images_dir.glob("*.jpg"))
    expected = len(json.loads(config.karpathy_json.read_text())["images"])
    if len(jpgs) < expected:
        raise ValueError(f"{len(jpgs)} images in {config.images_dir}, expected >= {expected} (partial download?)")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # jpgs already compressed
        for jpg in jpgs:
            zf.write(jpg, f"Images/{jpg.name}")
        zf.write(config.karpathy_json, "dataset_flickr8k.json")
        zf.write(config.vocab_path, "vocab.json")

    print(f"wrote {out} ({len(jpgs)} images + dataset_flickr8k.json + vocab.json)")
    return out


if __name__ == "__main__":
    make_train_zip()
