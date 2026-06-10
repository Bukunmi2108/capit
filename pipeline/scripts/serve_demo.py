"""Exit gate: knowing only a Hub repo id, download the artifact and caption one image.

This is the seed of backend/app.py — it proves capit-sat.pt is a self-sufficient inference
contract (no checkpoint, no training data, no hardcoded preprocess).
"""

from __future__ import annotations

import argparse
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from capit.serving import caption, load_artifact, make_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--repo-id", default="Bukunmi2108/capit-sat")
    args = parser.parse_args()

    artifact = hf_hub_download(args.repo_id, "capit-sat.pt")
    vocab_path = hf_hub_download(args.repo_id, "vocab.json")
    encoder, decoder, vocab, preprocess = load_artifact(artifact, vocab_path)
    transform = make_transform(preprocess)

    with Image.open(args.image) as img:
        tensor = transform((ImageOps.exif_transpose(img) or img).convert("RGB"))
    words, _, _ = caption(encoder, decoder, vocab, tensor)
    print(" ".join(words))


if __name__ == "__main__":
    main()
