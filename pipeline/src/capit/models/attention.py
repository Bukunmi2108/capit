"""Attention module for processing spatial features."""

from __future__ import annotations

import torch
from torch import nn
from capit.config import config

class BahdanauAttention(nn.Module):
    def __init__(
        self,
        encoder_dim: int = config.encoder_dim,
        decoder_dim: int = config.decoder_dim,
        attention_dim: int = config.attention_dim,
    ) -> None:
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        att1 = self.encoder_att(features)
        att2 = self.decoder_att(h).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha