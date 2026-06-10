"""Stage 2.2 — BahdanauAttention. Pure tensor tests; no data needed."""

import pytest
import torch

from capit.config import config
from capit.models.attention import BahdanauAttention

B = 3
L = config.encoded_size**2  # 196 locations


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(config.seed)


def _inputs(requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn(B, L, config.encoder_dim, requires_grad=requires_grad)
    h = torch.randn(B, config.decoder_dim, requires_grad=requires_grad)
    return features, h


def test_output_shapes():
    context, alpha = BahdanauAttention()(*_inputs())
    assert context.shape == (B, config.encoder_dim)
    assert alpha.shape == (B, L)


def test_alpha_sums_to_one():
    _, alpha = BahdanauAttention()(*_inputs())
    assert torch.allclose(alpha.sum(dim=1), torch.ones(B), atol=1e-5)


def test_alpha_is_a_distribution():
    _, alpha = BahdanauAttention()(*_inputs())
    assert (alpha >= 0).all()
    assert not torch.isnan(alpha).any()


def test_no_nan_in_context():
    context, _ = BahdanauAttention()(*_inputs())
    assert not torch.isnan(context).any()


def test_alpha_depends_on_hidden_state():
    att = BahdanauAttention()
    features, _ = _inputs()
    _, a1 = att(features, torch.randn(B, config.decoder_dim))
    _, a2 = att(features, torch.randn(B, config.decoder_dim))
    assert not torch.allclose(a1, a2)


def test_batch_independence():
    att = BahdanauAttention()
    features, h = _inputs()
    ctx_all, alpha_all = att(features, h)
    ctx_row, alpha_row = att(features[:1], h[:1])
    assert torch.allclose(alpha_all[:1], alpha_row, atol=1e-6)
    assert torch.allclose(ctx_all[:1], ctx_row, atol=1e-6)


def test_gradient_flows_to_both_inputs():
    features, h = _inputs(requires_grad=True)
    context, _ = BahdanauAttention()(features, h)
    context.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert h.grad is not None and h.grad.abs().sum() > 0
