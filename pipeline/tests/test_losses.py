"""Stage 2.3 — caption loss. Hand-built logits pin the shift, padding, and regularizer."""

import torch

from capit.config import config
from capit.losses import caption_loss, word_cross_entropy

V = 12


def _peaked(rows: list[list[int]]) -> torch.Tensor:
    logits = torch.full((1, len(rows[0]), V), -10.0)
    for t, tok in enumerate(rows[0]):
        logits[0, t, tok] = 10.0
    return logits


def test_cross_entropy_targets_are_shifted_by_one():
    captions = torch.tensor([[1, 5, 7, 2]])  # T=3, shifted targets = [5, 7, 2]
    assert word_cross_entropy(_peaked([[5, 7, 2]]), captions).item() < 0.01
    # predicting the inputs (captions[:, :-1]) instead of the shifted targets must score badly
    assert word_cross_entropy(_peaked([[1, 5, 7]]), captions).item() > 1.0


def test_pad_targets_are_ignored():
    captions = torch.tensor([[1, 5, 2, 0]])  # T=3, targets = [5, 2, 0]; position 2 is PAD
    logits = _peaked([[5, 2, 0]])
    base = word_cross_entropy(logits, captions)
    corrupted = logits.clone()
    corrupted[0, 2] = torch.randn(V) * 50  # corrupt the PAD-target position
    assert torch.isclose(base, word_cross_entropy(corrupted, captions))


def test_regularizer_is_alpha_c_on_zero_attention():
    captions = torch.tensor([[1, 5, 2, 0], [1, 6, 2, 0]])
    logits = torch.zeros(2, 3, V)
    alphas = torch.zeros(2, 3, 196)  # time-sum 0 per location → (1-0)^2 = 1
    reg = caption_loss(logits, alphas, captions) - word_cross_entropy(logits, captions)
    assert torch.isclose(reg, torch.tensor(config.alpha_c))
