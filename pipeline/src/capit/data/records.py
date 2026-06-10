"""Pure operations over Karpathy image records (the `images` list of the split JSON)."""

from __future__ import annotations

import random


def train_captions(records: list[dict]) -> list[list[str]]:
    captions = [s["tokens"] for r in records if r["split"] == "train" for s in r["sentences"]]
    if not captions:
        splits = sorted({r["split"] for r in records})
        raise ValueError(f"no train records found; available splits: {splits}")
    return captions


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
