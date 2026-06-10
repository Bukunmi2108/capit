"""Stage 2.3 — Decoder. Hermetic tests on random features + synthetic captions."""

import pytest
import torch

from capit.config import config
from capit.losses import caption_loss
from capit.models.decoder import Decoder

V = 12
B = 3
L = config.encoded_size**2

# captions sorted by length descending; PAD=0 START=1 END=2
CAPTIONS = torch.tensor(
    [
        [1, 5, 6, 7, 8, 2],
        [1, 5, 6, 2, 0, 0],
        [1, 7, 2, 0, 0, 0],
    ]
)
LENGTHS = torch.tensor([6, 4, 3])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(config.seed)


def _features() -> torch.Tensor:
    return torch.randn(B, L, config.encoder_dim)


def test_forward_shapes():
    logits, alphas, decode_lengths = Decoder(vocab_size=V)(_features(), CAPTIONS, LENGTHS)
    assert decode_lengths == [5, 3, 2]
    assert logits.shape == (B, 5, V)
    assert alphas.shape == (B, 5, L)


def test_no_nan():
    logits, alphas, _ = Decoder(vocab_size=V)(_features(), CAPTIONS, LENGTHS)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(alphas).any()


def test_shrinking_leaves_finished_rows_zero():
    logits, alphas, decode_lengths = Decoder(vocab_size=V)(_features(), CAPTIONS, LENGTHS)
    for i, dl in enumerate(decode_lengths):
        assert torch.all(logits[i, dl:] == 0)
        assert torch.all(alphas[i, dl:] == 0)


def test_padded_positions_contribute_zero_loss():
    logits, alphas, decode_lengths = Decoder(vocab_size=V)(_features(), CAPTIONS, LENGTHS)
    base = caption_loss(logits, alphas, CAPTIONS)
    corrupted = logits.clone()
    for i, dl in enumerate(decode_lengths):
        corrupted[i, dl:] = 999.0
    assert torch.isclose(base, caption_loss(corrupted, alphas, CAPTIONS))


def test_forward_rejects_unsorted_batch():
    captions = torch.tensor([[1, 7, 2, 0, 0, 0], [1, 5, 6, 7, 8, 2]])  # ascending lengths
    lengths = torch.tensor([3, 6])
    with pytest.raises(ValueError):
        Decoder(vocab_size=V)(torch.randn(2, L, config.encoder_dim), captions, lengths)


def test_gradient_reaches_decoder_params():
    dec = Decoder(vocab_size=V)
    logits, alphas, _ = dec(_features(), CAPTIONS, LENGTHS)
    caption_loss(logits, alphas, CAPTIONS).backward()
    for p in (dec.embedding.weight, dec.fc.weight, dec.f_beta.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_greedy_returns_tokens_and_restores_mode():
    dec = Decoder(vocab_size=V)
    dec.train()
    out = dec.greedy(_features(), start_id=1, end_id=2, max_len=10)
    assert len(out) == B
    assert all(isinstance(seq, list) and 2 not in seq for seq in out)
    assert dec.training
