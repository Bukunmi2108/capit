"""Overfit-one-batch harness (Stage 2.3, killer gate #1).

Trains the decoder on ONE batch of N distinct subsample images (encoder frozen, features
cached once) and reports whether it memorizes them. Trains on the full caption loss
(CE + doubly-stochastic regularizer); the meaningful signal is the CE term, since the
regularizer floors near ~0.9 for short captions so a sub-0.5 total is unreachable.
"""

from __future__ import annotations

import torch
from torch.optim import Adam

from capit.config import config
from capit.data.dataset import CaptionDataset, build_transform, collate_fn
from capit.data.vocab import END, START, Vocab
from capit.losses import caption_loss, word_cross_entropy
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder


def _distinct_image_batch(ds: CaptionDataset, n: int):
    seen: set[str] = set()
    picks = []
    for idx, (rec, _) in enumerate(ds.pairs):
        if rec["filename"] not in seen:
            seen.add(rec["filename"])
            picks.append(idx)
        if len(picks) == n:
            break
    if len(picks) < n:
        raise ValueError(f"requested {n} distinct images but dataset has only {len(picks)}")
    return collate_fn([ds[i] for i in picks])


def run_overfit(steps: int = 400, n_images: int = 4) -> dict:
    torch.manual_seed(config.seed)
    vocab = Vocab.load(config.vocab_path)
    ds = CaptionDataset(config.subsample_root, "train", vocab, build_transform())
    images, captions, lengths = _distinct_image_batch(ds, n_images)

    encoder = Encoder(pretrained=True)
    with torch.no_grad():
        features = encoder(images)  # frozen → cache once, never recompute

    decoder = Decoder(vocab_size=len(vocab))
    decoder.train()
    opt = Adam(decoder.parameters(), lr=config.decoder_lr)

    ce_curve = []
    for step in range(steps):
        logits, alphas, _ = decoder(features, captions, lengths)
        ce = word_cross_entropy(logits, captions)
        loss = caption_loss(logits, alphas, captions)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss diverged to {loss.item()} at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        ce_curve.append(ce.item())

    decoded = decoder.greedy(features, vocab.word2id[START], vocab.word2id[END])
    decoded_words = [vocab.decode(ids) for ids in decoded]
    target_words = [vocab.decode(captions[i, 1 : lengths[i] - 1].tolist()) for i in range(n_images)]
    return {"ce_curve": ce_curve, "final_ce": ce_curve[-1], "decoded": decoded_words, "targets": target_words}
