"""Export the serving artifact: best.pt (training checkpoint) -> capit-sat.pt (inference contract)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from capit.checkpoint import load
from capit.config import config
from capit.data.vocab import Vocab
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder
from capit.serving import build_artifact


def _split_counts(karpathy_json: Path) -> dict[str, int]:
    records = json.loads(karpathy_json.read_text())["images"]
    counts: dict[str, int] = {}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts


def _scores_table(metrics: dict[str, dict[str, float]]) -> str:
    cols = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"]
    lines = ["| beam | " + " | ".join(cols) + " |", "|-----:|" + "|".join(["-------:"] * len(cols)) + "|"]
    for beam in sorted(metrics, key=int):
        s = metrics[beam]
        lines.append(f"| {beam} | " + " | ".join(f"{s[c]:.2f}" for c in cols) + " |")
    return "\n".join(lines)


def _model_card(
    repo_id: str, splits: dict[str, int], best_bleu4: float, best_epoch: int, metrics: dict[str, dict[str, float]]
) -> str:
    return f"""---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: image-to-text
tags:
- image-captioning
- show-attend-and-tell
- visual-attention
datasets:
- flickr8k
metrics:
- bleu
- cider
---

# capit-sat

Show, Attend and Tell image captioner, trained from scratch on Flickr8k (Karpathy split).
The glass-box half of [capit](https://github.com/Bukunmi2108/capit) — exposes per-word
attention, beam candidates, and word-by-word playback.

## Test-set scores (pycocoevalcap, Karpathy test = {splits.get('test', '?')} images)

{_scores_table(metrics)}

## Training

- Backbone: frozen ResNet-50 (ImageNet). Decoder trained from scratch.
- Best val BLEU-4 {best_bleu4:.2f} at epoch {best_epoch} (early-stopped); Colab T4.
- Splits: train {splits.get('train', '?')}, val {splits.get('val', '?')}, test {splits.get('test', '?')}.

## Known limitation

Attention is effectively 7x7: ResNet-50 at 224px is natively 7x7 and the encoder upsamples
to 14x14, so heatmaps are coarse (~32px blocks). Captions are grounded; the spots are
region-level, not pixel-level.

## Use

`huggingface_hub.hf_hub_download("{repo_id}", "capit-sat.pt")` + `vocab.json`, then
`capit.serving.load_artifact(...)`.
"""


def export(ckpt_path: Path, vocab_path: Path, out_dir: Path, repo_id: str, metrics_json: Path) -> Path:
    if not metrics_json.exists():
        raise FileNotFoundError(
            f"metrics file {metrics_json} not found — generate it first:\n"
            f"  uv run python -m capit.evaluate --out-json {metrics_json}"
        )
    metrics = json.loads(metrics_json.read_text())
    vocab = Vocab.load(vocab_path)
    state = load(ckpt_path)
    if state.vocab_sha256 != vocab.sha256():
        raise ValueError(f"vocab mismatch: ckpt {state.vocab_sha256[:8]} != vocab {vocab.sha256()[:8]}")

    encoder = Encoder(pretrained=True)
    decoder = Decoder(vocab_size=len(vocab))
    decoder.load_state_dict(state.model_state)
    blob = build_artifact(encoder, decoder, vocab)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "capit-sat.pt"
    tmp = artifact_path.with_name(artifact_path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, artifact_path)
    shutil.copyfile(vocab_path, out_dir / "vocab.json")

    splits = _split_counts(config.karpathy_json)
    (out_dir / "README.md").write_text(_model_card(repo_id, splits, state.best_bleu4, state.best_epoch, metrics))
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(config.ckpt_dir / "best.pt"))
    parser.add_argument("--vocab", default=str(config.vocab_path))
    parser.add_argument("--out-dir", default=str(config.data_root / "artifact"))
    parser.add_argument("--repo-id", default="Bukunmi2108/capit-sat")
    parser.add_argument("--metrics-json", default=str(config.data_root / "eval_results.json"))
    args = parser.parse_args()

    path = export(Path(args.ckpt), Path(args.vocab), Path(args.out_dir), args.repo_id, Path(args.metrics_json))
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({size_mb:.1f} MB), vocab.json, README.md to {args.out_dir}")
    print(f"push: hf upload {args.repo_id} {args.out_dir} . --repo-type model")


if __name__ == "__main__":
    main()
