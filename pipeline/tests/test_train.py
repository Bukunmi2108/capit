"""Stage 3.2 — training loop.

corpus_bleu4 is hermetic and always runs; the full-loop gate is opt-in (RUN_TRAIN=1)
and needs the subsample + vocab.
"""

import os

import pytest
from torch import nn
from torch.optim import Adam

from capit.checkpoint import load, save
from capit.config import config
from capit.train import corpus_bleu4, train

_GATE = pytest.mark.skipif(
    os.environ.get("RUN_TRAIN") != "1"
    or not (config.subsample_json.is_file() and config.vocab_path.is_file()),
    reason="slow gate — set RUN_TRAIN=1 (needs subsample + vocab)",
)


def test_corpus_bleu4_perfect_and_disjoint():
    refs = {0: [["a", "dog", "runs", "in", "snow"]]}
    assert corpus_bleu4(refs, {0: ["a", "dog", "runs", "in", "snow"]}) == pytest.approx(100.0)
    assert corpus_bleu4(refs, {0: ["totally", "different", "words", "here", "now"]}) < 10.0


def test_corpus_bleu4_empty_references_raises():
    with pytest.raises(ValueError):
        corpus_bleu4({}, {})


@_GATE
def test_train_runs_checkpoints_and_resumes(tmp_path):
    ckpt = tmp_path / "ckpt"
    train(config.subsample_root, ckpt, resume="none", max_epochs=2, num_workers=0)
    assert (ckpt / "latest.pt").is_file()
    assert (ckpt / "best.pt").is_file()
    assert load(ckpt / "latest.pt").epoch == 1

    train(config.subsample_root, ckpt, resume="auto", max_epochs=4, num_workers=0)
    resumed = load(ckpt / "latest.pt")
    assert resumed.epoch == 3  # continued past the checkpointed epoch 1
    assert resumed.optim_state


@_GATE
def test_resume_rejects_vocab_mismatch(tmp_path):
    ckpt = tmp_path / "ckpt"
    dummy = nn.Linear(1, 1)
    save(ckpt / "latest.pt", dummy, Adam(dummy.parameters()), 0, 0.0, 0, "wrong-hash")
    with pytest.raises(ValueError):
        train(config.subsample_root, ckpt, resume="auto", max_epochs=1, num_workers=0)
