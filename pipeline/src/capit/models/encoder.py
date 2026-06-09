"""Encoder: a frozen ResNet-50 backbone producing [B, 196, 2048] spatial features."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from capit.config import config


class Encoder(nn.Module):
    def __init__(self, encoded_size: int = config.encoded_size, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.backbone(images))
        return x.flatten(2).permute(0, 2, 1)

    def fine_tune(self, blocks: tuple[int, ...] = ()) -> None:
        for i in blocks:
            for p in self.backbone[i].parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "Encoder":
        super().train(mode)
        self.backbone.eval()
        return self
