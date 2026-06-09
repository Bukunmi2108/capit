"""Stage 0.2 exit gate — dataset integrity.

The whole module is skipped when the data isn't on disk, so CI never needs the dataset.
When present, it verifies the counts the rest of the pipeline relies on.
"""

import json
from collections import Counter
import pytest
from PIL import Image
from capit.config import config

EXPECTED_JPGS = 8091
EXPECTED_IMAGE_RECORDS = 8000
EXPECTED_CAPTIONS_PER_IMAGE = 5
EXPECTED_SPLITS = {"train": 6000, "val": 1000, "test": 1000}


def _data_present() -> bool:
    return config.images_dir.is_dir() and config.karpathy_json.is_file()


pytestmark = pytest.mark.skipif(
    not _data_present(),
    reason="Flickr8k data not on disk (run: python -m capit.data.download)",
)


@pytest.fixture(scope="module")
def karpathy() -> dict:
    with open(config.karpathy_json) as f:
        return json.load(f)


def test_dataset_integrity(karpathy: dict) -> None:
    # 8,091 images on disk (Kaggle mirror).
    jpgs = list(config.images_dir.glob("*.jpg"))
    assert len(jpgs) == EXPECTED_JPGS, f"expected {EXPECTED_JPGS} jpgs, found {len(jpgs)}"

    # The JSON is the source of truth: 8,000 records, exactly 5 captions each.
    images = karpathy["images"]
    assert len(images) == EXPECTED_IMAGE_RECORDS
    assert all(len(img["sentences"]) == EXPECTED_CAPTIONS_PER_IMAGE for img in images)

    # Canonical Karpathy split (flickr8k has only train/val/test — no restval).
    splits = dict(Counter(img["split"] for img in images))
    assert splits == EXPECTED_SPLITS, f"split tally {splits} != {EXPECTED_SPLITS}"

    # Every referenced image exists on disk; the ~91 unreferenced Kaggle images may not.
    disk = {p.name for p in jpgs}
    missing = [img["filename"] for img in images if img["filename"] not in disk]
    assert not missing, f"{len(missing)} referenced images absent, e.g. {missing[:3]}"


def test_spot_check_images_open(karpathy: dict) -> None:
    # 3 images open with PIL and their captions read sensibly.
    for img in karpathy["images"][:3]:
        path = config.images_dir / img["filename"]
        with Image.open(path) as im:
            im.load()
        caption = " ".join(img["sentences"][0]["tokens"])
        assert len(caption.split()) >= 3, f"caption too short: {caption!r}"
