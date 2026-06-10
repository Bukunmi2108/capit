"""Stage 2.3 exit gate — overfit one batch (killer gate #1).

Slow (~3 min): opt-in via RUN_OVERFIT=1, and needs the subsample + vocab. Otherwise the
gate lives in scripts/overfit_one_batch.py (run + eyeball the decoded captions).
"""

import os

import pytest

from capit.config import config
from capit.overfit import run_overfit

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OVERFIT") != "1"
    or not (config.subsample_json.is_file() and config.vocab_path.is_file()),
    reason="slow gate — set RUN_OVERFIT=1 (needs subsample + vocab)",
)


def test_overfit_one_batch():
    result = run_overfit(steps=200)
    assert result["final_ce"] < 0.5, f"ce {result['final_ce']:.3f}: model failed to memorize"
    matches = sum(d == t for d, t in zip(result["decoded"], result["targets"]))
    assert matches == len(result["targets"]), f"{matches}/{len(result['targets'])} captions reproduced"
