"""Stage 4.1 — greedy + beam decoding. Hermetic: random decoder + random features."""

import pytest
import torch

from capit.config import config
from capit.decode import beam_search, greedy
from capit.models.decoder import Decoder

V = 30
L = config.encoded_size**2
START, END = 1, 2


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(config.seed)


def _features() -> torch.Tensor:
    return torch.randn(1, L, config.encoder_dim)


def test_beam_k1_equals_greedy():
    dec = Decoder(vocab_size=V)
    feats = _features()
    g_ids, _ = greedy(dec, feats, START, END)
    b_ids, _, _ = beam_search(dec, feats, START, END, k=1)
    assert b_ids == g_ids


def test_greedy_alphas_align_with_ids():
    dec = Decoder(vocab_size=V)
    ids, alphas = greedy(dec, _features(), START, END)
    assert alphas.shape == (len(ids), L)
    assert not torch.isnan(alphas).any()


def test_beam_k3_caption_and_road_not_taken():
    dec = Decoder(vocab_size=V)
    ids, alphas, beams = beam_search(dec, _features(), START, END, k=3)
    assert alphas.shape == (len(ids), L)
    assert beams[0][0] == ids  # winner first
    assert all(isinstance(s, float) for _, s in beams)
    assert END not in ids  # specials stripped
