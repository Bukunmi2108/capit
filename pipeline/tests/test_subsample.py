"""Dev subsample.

Pure selection tests always run; structural checks on the on-disk artifact skip when
it hasn't been generated.
"""

import json
from collections import Counter

import pytest

from capit.config import config
from capit.data.records import select_records


def _records(split: str, n: int) -> list[dict]:
    return [{"filename": f"{split}_{i:03d}.jpg", "split": split, "imgid": i} for i in range(n)]


SYNTH = _records("train", 20) + _records("val", 8) + _records("test", 8)
COUNTS = {"train": 5, "val": 3, "test": 2}


def test_select_counts_and_splits():
    sel = select_records(SYNTH, COUNTS, seed=13)
    assert len(sel) == 10
    assert Counter(r["split"] for r in sel) == {"train": 5, "val": 3, "test": 2}


def test_select_is_deterministic():
    assert select_records(SYNTH, COUNTS, 13) == select_records(SYNTH, COUNTS, 13)
    assert select_records(SYNTH, COUNTS, 13) != select_records(SYNTH, COUNTS, 14)


def test_select_records_are_verbatim():
    sel = select_records(SYNTH, COUNTS, 13)
    assert all(r in SYNTH for r in sel)


def test_select_raises_when_pool_too_small():
    with pytest.raises(ValueError):
        select_records(SYNTH, {"train": 999}, 13)


def _artifact_present() -> bool:
    return config.subsample_json.is_file() and config.subsample_images_dir.is_dir()


pytestmark_structural = pytest.mark.skipif(
    not _artifact_present(),
    reason="dev subsample not generated (run: python pipeline/scripts/make_subsample.py)",
)


@pytestmark_structural
def test_artifact_structure():
    sub = json.loads(config.subsample_json.read_text())
    images = sub["images"]
    assert len(images) == sum(config.subsample_counts.values())
    assert Counter(r["split"] for r in images) == config.subsample_counts

    jpgs = {p.name for p in config.subsample_images_dir.glob("*.jpg")}
    assert len(jpgs) == len(images)
    assert all(r["filename"] in jpgs for r in images)

    assert sub["dataset"] == json.loads(config.karpathy_json.read_text())["dataset"]


@pytestmark_structural
def test_artifact_records_are_verbatim():
    sub = json.loads(config.subsample_json.read_text())
    full = {r["filename"]: r for r in json.loads(config.karpathy_json.read_text())["images"]}
    assert all(r == full[r["filename"]] for r in sub["images"])
