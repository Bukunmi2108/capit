"""Greedy and beam-search decoding with per-word attention.

Both operate on a single image's encoder features ``[1, L, encoder_dim]`` and return the caption token ids (specials stripped) plus an aligned ``[T, L]`` alpha tensor — the per-word attention that drives the heatmaps. beam_search also returns the full completed set as ``(tokens, normalized_score)`` pairs, winner first ("the road not taken").
"""

from __future__ import annotations

import torch
from torch.nn.functional import log_softmax
from capit.config import config
from capit.models.decoder import Decoder


@torch.no_grad()
def greedy(
    decoder: Decoder, features: torch.Tensor, start_id: int, end_id: int, max_len: int = config.max_decode_len
) -> tuple[list[int], torch.Tensor]:
    decoder.eval()
    h, c = decoder.init_hidden(features)
    token = features.new_full((1,), start_id, dtype=torch.long)
    ids: list[int] = []
    alphas: list[torch.Tensor] = []
    for _ in range(max_len):
        logits, alpha, h, c = decoder.step(token, h, c, features)
        token = logits.argmax(dim=1)
        if int(token) == end_id:
            break
        ids.append(int(token))
        alphas.append(alpha[0])
    stacked = torch.stack(alphas) if alphas else features.new_zeros(0, features.size(1))
    return ids, stacked


@torch.no_grad()
def beam_search(
    decoder: Decoder,
    features: torch.Tensor,
    start_id: int,
    end_id: int,
    k: int = config.beam_k,
    max_len: int = config.max_decode_len,
    alpha_norm: float = config.beam_alpha,
) -> tuple[list[int], torch.Tensor, list[tuple[list[int], float]]]:
    decoder.eval()
    vocab_size, num_pixels = decoder.fc.out_features, features.size(1)
    feats = features.expand(k, num_pixels, -1)
    h, c = decoder.init_hidden(feats)
    seqs = features.new_full((k, 1), start_id, dtype=torch.long)
    seq_alphas = features.new_zeros(k, 1, num_pixels)
    scores = features.new_zeros(k, 1)
    completed: list[tuple[list[int], torch.Tensor, float]] = []

    for step in range(max_len):
        logits, alpha, h, c = decoder.step(seqs[:, -1], h, c, feats)
        logp = scores + log_softmax(logits, dim=1)
        flat = logp[0] if step == 0 else logp.reshape(-1)  # step 0: all beams identical
        top_scores, top_idx = flat.topk(seqs.size(0))
        beam_idx, word_idx = top_idx // vocab_size, top_idx % vocab_size

        seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
        seq_alphas = torch.cat([seq_alphas[beam_idx], alpha[beam_idx].unsqueeze(1)], dim=1)
        scores, h, c, feats = top_scores.unsqueeze(1), h[beam_idx], c[beam_idx], feats[beam_idx]

        ended = word_idx == end_id
        for i in ended.nonzero().flatten().tolist():
            completed.append((seqs[i, 1:-1].tolist(), seq_alphas[i, 1:-1], float(scores[i])))
        keep = (~ended).nonzero().flatten()
        if len(keep) == 0:
            break
        seqs, seq_alphas, scores = seqs[keep], seq_alphas[keep], scores[keep]
        h, c, feats = h[keep], c[keep], feats[keep]

    if not completed:  # nothing emitted <end> by max_len → fall back to the best live hypothesis
        i = int(scores.argmax())
        completed.append((seqs[i, 1:].tolist(), seq_alphas[i, 1:], float(scores[i])))

    ranked = sorted(completed, key=lambda x: x[2] / max(len(x[0]), 1) ** alpha_norm, reverse=True)
    beams = [(toks, score / max(len(toks), 1) ** alpha_norm) for toks, _, score in ranked]
    win_tokens, win_alphas, _ = ranked[0]
    return win_tokens, win_alphas, beams
