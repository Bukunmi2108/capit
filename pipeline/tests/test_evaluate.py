"""Stage 4.2 — official BLEU/CIDEr harness. Hermetic scorer sanity + opt-in eval-on-test gate."""

import os

import pytest

from capit.config import config
from capit.evaluate import evaluate, score


def test_score_perfect_match():
    refs = {
        0: [["a", "brown", "dog", "runs", "fast"], ["the", "dog", "runs", "very", "fast"]],
        1: [["a", "small", "cat", "sleeps", "quietly"]],
    }
    cands = {0: ["a", "brown", "dog", "runs", "fast"], 1: ["a", "small", "cat", "sleeps", "quietly"]}
    s = score(refs, cands)
    assert s["BLEU-4"] == pytest.approx(100.0, abs=1e-6)
    assert s["BLEU-1"] == pytest.approx(100.0, abs=1e-6)


def test_score_disjoint():
    refs = {0: [["a", "brown", "dog", "runs", "fast"]]}
    cands = {0: ["xyz", "qpr", "lmn", "uvw", "abc"]}
    s = score(refs, cands)
    assert s["BLEU-1"] == pytest.approx(0.0, abs=1e-6)
    assert s["BLEU-4"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    os.environ.get("RUN_EVAL") != "1" or not (config.ckpt_dir / "best.pt").exists(),
    reason="set RUN_EVAL=1 and provide data/checkpoints/best.pt to run the test-set gate",
)
def test_eval_on_test_meets_gate():
    # pycocoevalcap corpus BLEU (unsmoothed) != train.corpus_bleu4 (NLTK smoothed); not directly comparable to best_bleu4.
    results = evaluate(config.flickr8k_dir, config.ckpt_dir / "best.pt", beams=(3,))
    assert results[3]["BLEU-4"] >= 16.0
