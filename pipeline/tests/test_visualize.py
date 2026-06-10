"""Stage 4.3 — attention overlay plumbing. Hermetic: synthetic alphas, no model or checkpoint."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from visualize_attention import heatmap, overlay_grid

from capit.config import config

L = config.encoded_size**2


def test_heatmap_upscales_grid_to_crop():
    heat = heatmap(torch.rand(L))
    assert heat.shape == (config.crop, config.crop)
    assert not torch.isnan(torch.from_numpy(heat)).any()


def test_overlay_grid_has_panel_per_word_plus_input():
    words = ["a", "dog", "runs"]
    fig = overlay_grid(torch.rand(3, config.crop, config.crop), words, torch.rand(len(words), L))
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert labels == ["input", *words]
    plt.close(fig)
