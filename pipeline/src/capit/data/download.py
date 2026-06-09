"""Dataset acquisition.

Two independent, idempotent fetchers that materialize Flickr8k into the canonical ``data/`` root so the directory is self-contained .

The Karpathy JSON is the source of truth downstream; the ~91 Kaggle images it does not
reference are never loaded, and captions.txt is downloaded for completeness but unused.
"""

from __future__ import annotations

import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from capit.config import config

KARPATHY_ZIP_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)
KAGGLE_DATASET = "adityajn105/flickr8k"
_HTTP_TIMEOUT = 60


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def download_flickr8k(dest: Path | None = None) -> Path:
    """Materialize Flickr8k images + captions.txt into ``dest`` (default ``config.flickr8k_dir``)."""
    dest = Path(dest) if dest is not None else config.flickr8k_dir
    images_dir = dest / "Images"
    captions = dest / "captions.txt"
    sentinel = dest / ".complete"

    if sentinel.exists():
        return dest

    import kagglehub  # lazy: keeps the module importable without the dep installed

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src_images = src / "Images"
    if not src_images.is_dir():
        contents = sorted(p.name for p in src.iterdir())
        raise FileNotFoundError(f"expected Images/ under {src}, found: {contents}")
    src_captions = src / "captions.txt"
    if not src_captions.is_file():
        contents = sorted(p.name for p in src.iterdir())
        raise FileNotFoundError(f"expected captions.txt under {src}, found: {contents}")

    images_dir.mkdir(parents=True, exist_ok=True)
    for jpg in src_images.glob("*.jpg"):
        target = images_dir / jpg.name
        if not target.exists():
            _copy_atomic(jpg, target)
    _copy_atomic(src_captions, captions)

    sentinel.touch()
    return dest


def download_karpathy_splits(dest: Path | None = None) -> Path:
    """Download Karpathy's caption_datasets.zip and extract dataset_flickr8k.json into ``dest``.
    """
    dest = Path(dest) if dest is not None else config.karpathy_dir
    json_path = dest / "dataset_flickr8k.json"
    if json_path.is_file():
        return json_path

    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(KARPATHY_ZIP_URL, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 — trusted https URL
        archive = resp.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        members = [n for n in zf.namelist() if n.endswith("dataset_flickr8k.json")]
        if not members:
            raise FileNotFoundError(
                f"dataset_flickr8k.json not in archive; contents: {zf.namelist()}"
            )
        tmp = json_path.with_name(json_path.name + ".tmp")
        try:
            with zf.open(members[0]) as f_in, open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.replace(tmp, json_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    return json_path


def main() -> None:
    flickr_dir = download_flickr8k()
    json_path = download_karpathy_splits()
    n_jpg = len(list((flickr_dir / "Images").glob("*.jpg")))
    print(f"flickr8k: {n_jpg} jpgs in {flickr_dir / 'Images'}")
    print(f"karpathy: {json_path}")


if __name__ == "__main__":
    main()
