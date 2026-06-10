"""Training checkpoints: atomic save/load of model + optimizer state for resumable runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainState:
    model_state: dict[str, Any]
    optim_state: dict[str, Any]
    epoch: int
    best_bleu4: float
    best_epoch: int
    vocab_sha256: str


def save(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_bleu4: float,
    best_epoch: int,
    vocab_sha256: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_bleu4": best_bleu4,
        "best_epoch": best_epoch,
        "vocab_sha256": vocab_sha256,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: Path | str) -> TrainState:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    return TrainState(
        model_state=state["model"],
        optim_state=state["optimizer"],
        epoch=state["epoch"],
        best_bleu4=state["best_bleu4"],
        best_epoch=state.get("best_epoch", state["epoch"]),
        vocab_sha256=state["vocab_sha256"],
    )
