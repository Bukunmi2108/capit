"""Decode one image at beam 3 and render a per-word attention grid over it."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.transform import pyramid_expand
from torchvision import transforms

from capit.checkpoint import load
from capit.config import Config, config
from capit.data.dataset import CaptionDataset, build_transform
from capit.data.vocab import END, START, Vocab
from capit.decode import beam_search
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder


def display_transform(cfg: Config = config):
    return transforms.Compose(
        [transforms.Resize(cfg.resize), transforms.CenterCrop(cfg.crop), transforms.ToTensor()]
    )


def load_image(path: Path | str, transform) -> torch.Tensor:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img) or img
        return transform(img.convert("RGB"))


def heatmap(alpha: torch.Tensor, upscale: int = 16, sigma: float = 8.0) -> np.ndarray:
    grid = config.encoded_size
    return pyramid_expand(alpha.reshape(grid, grid).numpy(), upscale=upscale, sigma=sigma)


def _label(ax, text: str) -> None:
    ax.text(
        0.03, 0.95, text, transform=ax.transAxes, fontsize=11, va="top",
        color="black", bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )


def overlay_grid(display_img: torch.Tensor, words: list[str], alphas: torch.Tensor, attn_alpha: float = 0.8):
    base = display_img.permute(1, 2, 0).numpy()
    heats = [heatmap(a) for a in alphas]
    n = len(words) + 1
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False, facecolor="white")
    axes = axes.reshape(-1)
    axes[0].imshow(base)
    _label(axes[0], "input")
    for i, (word, heat) in enumerate(zip(words, heats), start=1):
        axes[i].imshow(base)
        axes[i].imshow(heat, alpha=attn_alpha, cmap="Greys_r")
        _label(axes[i], word)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


@torch.no_grad()
def visualize(image_path: Path | str, ckpt_path: Path | str, out_path: Path | str, k: int = config.beam_k) -> list[str]:
    vocab = Vocab.load(config.vocab_path)
    encoder = Encoder(pretrained=True).eval()
    decoder = Decoder(vocab_size=len(vocab))
    state = load(ckpt_path)
    if state.vocab_sha256 != vocab.sha256():
        raise ValueError(f"vocab mismatch: ckpt {state.vocab_sha256[:8]} != vocab {vocab.sha256()[:8]}")
    decoder.load_state_dict(state.model_state)
    decoder.eval()

    features = encoder(load_image(image_path, build_transform()).unsqueeze(0))
    tokens, alphas, _ = beam_search(decoder, features, vocab.word2id[START], vocab.word2id[END], k=k)
    words = vocab.decode(tokens)

    fig = overlay_grid(load_image(image_path, display_transform()), words, alphas)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return words


def _resolve_image(args) -> Path:
    if args.image:
        return Path(args.image)
    ds = CaptionDataset(args.data_root, "test", Vocab.load(config.vocab_path), build_transform())
    return Path(args.data_root) / "Images" / ds.records[args.index]["filename"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="explicit image path; overrides --index")
    parser.add_argument("--index", type=int, default=0, help="index into the test split")
    parser.add_argument("--data-root", default=str(config.flickr8k_dir))
    parser.add_argument("--ckpt", default=str(config.ckpt_dir / "best.pt"))
    parser.add_argument("--out", default=str(config.data_root / "attention.png"))
    args = parser.parse_args()

    image_path = _resolve_image(args)
    words = visualize(image_path, args.ckpt, args.out)
    print(f"{image_path.name}: {' '.join(words)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
