"""Caption loss: cross-entropy over real positions + doubly-stochastic attention regularizer."""

from __future__ import annotations

import torch
from torch.nn.functional import cross_entropy

from capit.config import config


def word_cross_entropy(logits: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    targets = captions[:, 1 : 1 + logits.size(1)]
    return cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)


def caption_loss(
    logits: torch.Tensor,
    alphas: torch.Tensor,
    captions: torch.Tensor,
    alpha_c: float = config.alpha_c,
) -> torch.Tensor:
    reg = alpha_c * ((1 - alphas.sum(dim=1)) ** 2).mean()
    return word_cross_entropy(logits, captions) + reg
