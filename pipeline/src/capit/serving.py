"""Serving artifact: build / load the self-contained inference contract (capit-sat.pt).

The artifact carries everything inference needs (encoder + decoder weights, model dims, the
exact preprocess spec, and the vocab hash) and nothing training-only. The backend rebuilds the
model from this alone — it never imports training code or touches a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

from capit.config import config
from capit.data.vocab import END, START, Vocab
from capit.decode import beam_search
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder


def build_artifact(encoder: Encoder, decoder: Decoder, vocab: Vocab) -> dict[str, Any]:
    return {
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "config": {
            "vocab_size": len(vocab),
            "encoder_dim": config.encoder_dim,
            "decoder_dim": config.decoder_dim,
            "attention_dim": config.attention_dim,
            "embed_dim": config.embed_dim,
            "encoded_size": config.encoded_size,
            "dropout": config.dropout,
            "beam_k": config.beam_k,
            "beam_alpha": config.beam_alpha,
            "max_decode_len": config.max_decode_len,
        },
        "preprocess": {
            "resize": config.resize,
            "crop": config.crop,
            "mean": list(config.imagenet_mean),
            "std": list(config.imagenet_std),
        },
        "vocab_sha256": vocab.sha256(),
    }


def make_transform(preprocess: dict[str, Any]):
    return transforms.Compose(
        [
            transforms.Resize(preprocess["resize"]),
            transforms.CenterCrop(preprocess["crop"]),
            transforms.ToTensor(),
            transforms.Normalize(preprocess["mean"], preprocess["std"]),
        ]
    )


def load_artifact(artifact_path: Path | str, vocab_path: Path | str):
    blob = torch.load(artifact_path, map_location="cpu", weights_only=True)
    vocab = Vocab.load(vocab_path)
    if vocab.sha256() != blob["vocab_sha256"]:
        raise ValueError(f"vocab mismatch: vocab {vocab.sha256()[:8]} != artifact {blob['vocab_sha256'][:8]}")
    cfg = blob["config"]
    encoder = Encoder(encoded_size=cfg["encoded_size"], pretrained=False)
    encoder.load_state_dict(blob["encoder_state"])
    encoder.eval()
    decoder = Decoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        decoder_dim=cfg["decoder_dim"],
        attention_dim=cfg["attention_dim"],
        dropout=cfg["dropout"],
    )
    decoder.load_state_dict(blob["decoder_state"])
    decoder.eval()
    return encoder, decoder, vocab, blob["preprocess"]


@torch.no_grad()
def caption(encoder: Encoder, decoder: Decoder, vocab: Vocab, image: torch.Tensor, k: int = config.beam_k):
    features = encoder(image.unsqueeze(0))
    tokens, alphas, beams = beam_search(decoder, features, vocab.word2id[START], vocab.word2id[END], k=k)
    return vocab.decode(tokens), alphas, beams
