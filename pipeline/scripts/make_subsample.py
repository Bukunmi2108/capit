"""Build the dev subsample: a small seeded slice of Flickr8k written in the same
on-disk format (Karpathy JSON + Images/) as the full set."""

from __future__ import annotations

import json
import random
import shutil
from capit.config import config
from capit.data.download import _copy_atomic


def select_records(records: list[dict], counts: dict[str, int], seed: int) -> list[dict]:
    by_split: dict[str, list[dict]] = {split: [] for split in counts}
    for rec in records:
        if rec["split"] in by_split:
            by_split[rec["split"]].append(rec)

    rng = random.Random(seed)
    selected: list[dict] = []
    for split, count in counts.items():
        pool = sorted(by_split[split], key=lambda r: r["filename"])
        if len(pool) < count:
            raise ValueError(f"{split}: need {count}, only {len(pool)} available")
        selected.extend(rng.sample(pool, count))
    return selected


def make_subsample() -> None:
    full = json.loads(config.karpathy_json.read_text())
    selected = select_records(full["images"], config.subsample_counts, config.seed)

    sources = [(config.images_dir / r["filename"], r) for r in selected]
    missing = [r["filename"] for src, r in sources if not src.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source images missing under {config.images_dir} "
            f"(rerun the Stage 0.2 download), e.g. {missing[:3]}"
        )

    if config.subsample_root.exists():
        shutil.rmtree(config.subsample_root)
    config.subsample_images_dir.mkdir(parents=True, exist_ok=True)
    for src, rec in sources:
        _copy_atomic(src, config.subsample_images_dir / rec["filename"])

    config.subsample_json.write_text(json.dumps({"images": selected, "dataset": full["dataset"]}))

    tally = {s: sum(r["split"] == s for r in selected) for s in config.subsample_counts}
    print(f"subsample: {len(selected)} images {tally} -> {config.subsample_root}")


if __name__ == "__main__":
    make_subsample()
