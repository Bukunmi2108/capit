"""Encoder.

Structural tests run offline (random weights). The non-degenerate-features test needs
ImageNet weights + a real subsample image and skips when either is absent.
"""

import pytest
import torch
from capit.config import config
from capit.models.encoder import Encoder


@pytest.fixture(scope="module")
def encoder() -> Encoder:
    return Encoder(pretrained=False)


def test_output_shape(encoder: Encoder):
    feats = encoder(torch.randn(2, 3, config.crop, config.crop))
    assert feats.shape == (2, config.encoded_size**2, config.encoder_dim)


def test_all_params_frozen(encoder: Encoder):
    total = sum(p.numel() for p in encoder.parameters())
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    assert total > 0
    assert frozen == total


def test_no_nan_on_random_input(encoder: Encoder):
    feats = encoder(torch.randn(2, 3, config.crop, config.crop))
    assert not torch.isnan(feats).any()


def test_eval_mode_determinism(encoder: Encoder):
    encoder.eval()
    x = torch.randn(1, 3, config.crop, config.crop)
    with torch.no_grad():
        assert torch.equal(encoder(x), encoder(x))


def test_eval_uses_running_stats_not_batch(encoder: Encoder):
    encoder.eval()
    x = torch.randn(1, 3, config.crop, config.crop)
    others = torch.randn(3, 3, config.crop, config.crop)
    with torch.no_grad():
        alone = encoder(x)
        batched = encoder(torch.cat([x, others]))[:1]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_train_keeps_backbone_in_eval(encoder: Encoder):
    encoder.train()
    assert not encoder.backbone.training
    encoder.eval()


def test_fine_tune_default_keeps_all_frozen():
    enc = Encoder(pretrained=False)
    enc.fine_tune()
    assert all(not p.requires_grad for p in enc.parameters())


def test_fine_tune_unfreezes_exactly_that_block():
    enc = Encoder(pretrained=False)
    enc.fine_tune((7,))  # index 7 = layer4
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    layer4 = sum(p.numel() for p in enc.backbone[7].parameters())
    assert trainable == layer4 > 0


@pytest.mark.skipif(
    not (config.subsample_json.is_file() and config.vocab_path.is_file()),
    reason="dev subsample or vocab not built",
)
def test_real_image_features_non_degenerate():
    from capit.data.dataset import CaptionDataset, build_transform
    from capit.data.vocab import Vocab

    vocab = Vocab.load(config.vocab_path)
    ds = CaptionDataset(config.subsample_root, "test", vocab, build_transform())
    _, image = next(ds.iter_images())
    encoder = Encoder(pretrained=True)
    with torch.no_grad():
        feats = encoder(image.unsqueeze(0))
    assert not torch.isnan(feats).any()
    assert feats.std().item() > 0
