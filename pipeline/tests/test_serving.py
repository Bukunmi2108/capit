"""Stage 4.4 — serving artifact round-trip. Hermetic: random encoder/decoder, no checkpoint/network."""

import json

import pytest
import torch
from PIL import Image

from capit.data.vocab import SPECIALS, Vocab
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder
from capit.serving import build_artifact, caption, load_artifact, make_transform


@pytest.fixture
def artifact(tmp_path):
    torch.manual_seed(0)
    vocab = Vocab(SPECIALS + ["a", "dog", "runs", "fast"])
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)
    encoder = Encoder(pretrained=False).eval()
    decoder = Decoder(vocab_size=len(vocab))
    blob = build_artifact(encoder, decoder, vocab)
    path = tmp_path / "capit-sat.pt"
    torch.save(blob, path)
    return path, vocab_path, encoder, decoder, vocab


def test_round_trip_reproduces_caption(artifact):
    path, vocab_path, encoder, decoder, vocab = artifact
    enc2, dec2, vocab2, preprocess = load_artifact(path, vocab_path)
    assert vocab2.id2word == vocab.id2word
    assert {"resize", "crop", "mean", "std"} <= preprocess.keys()
    torch.manual_seed(1)
    image = torch.rand(3, preprocess["crop"], preprocess["crop"])
    words, alphas, _ = caption(enc2, dec2, vocab2, image)
    ref_words, ref_alphas, _ = caption(encoder, decoder, vocab, image)
    assert words == ref_words
    assert torch.equal(alphas, ref_alphas)


def test_vocab_guard_raises_on_mismatch(artifact, tmp_path):
    path, vocab_path, *_ = artifact
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(SPECIALS + ["different", "words", "entirely"]))
    with pytest.raises(ValueError, match="vocab mismatch"):
        load_artifact(path, wrong)


def test_make_transform_shapes_image():
    t = make_transform({"resize": 256, "crop": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})
    out = t(Image.new("RGB", (400, 300)))
    assert out.shape == (3, 224, 224)
