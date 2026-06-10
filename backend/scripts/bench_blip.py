"""Benchmark BLIP caption latency + peak memory on CPU (Phase 5, killer gate #3).

No published CPU latency exists for blip-image-captioning-base; this measures it directly to
decide whether it ships as-is on the 2-vCPU HF Space, needs int8 quantization, or a fallback.
Standalone — no capit import; the backend is its own component.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL = "Salesforce/blip-image-captioning-base"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _test_images(data_root: Path, n: int) -> list[Path]:
    records = json.loads((data_root / "dataset_flickr8k.json").read_text())["images"]
    test = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["filename"])
    return [data_root / "Images" / r["filename"] for r in test[:n]]


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is KB on Linux


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@torch.no_grad()
def bench(processor, model, paths: list[Path], num_beams: int) -> dict:
    latencies: list[float] = []
    sample = ""
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            image = (ImageOps.exif_transpose(im) or im).convert("RGB")
        t0 = time.perf_counter()
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, num_beams=num_beams, max_new_tokens=30)
        dt = time.perf_counter() - t0
        if i == 0:  # warm-up: first call pays lazy init / caching
            sample = processor.decode(out[0], skip_special_tokens=True)
            continue
        latencies.append(dt)
    return {
        "num_beams": num_beams,
        "n": len(latencies),
        "mean_s": sum(latencies) / len(latencies),
        "p95_s": _pct(latencies, 0.95),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "sample_caption": sample,
    }


def _verdict(mean_s: float) -> str:
    if mean_s <= 5.0:
        return "SHIP AS-IS (comfortable margin under the 10s Space budget)"
    if mean_s <= 10.0:
        return "TIGHT — under budget locally but little Space margin; consider int8 quantization"
    return "MISS — exceeds budget; int8 quantize (OpenVINO/NNCF) then re-bench, else ViT-GPT2 fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "flickr8k"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2, help="2 simulates the HF Space's 2 vCPU")
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--out-json", default=str(_REPO_ROOT / "data" / "blip_bench.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL).eval()
    paths = _test_images(Path(args.data_root), args.n + 1)  # +1 for the discarded warm-up

    runs = [bench(processor, model, paths, b) for b in args.beams]
    peak_rss = _peak_rss_mb()
    headline = max(runs, key=lambda r: r["num_beams"])  # the slowest config is what the UI serves

    report = {
        "model": MODEL,
        "threads": args.threads,
        "torch_threads_available": torch.get_num_threads(),
        "peak_rss_mb": round(peak_rss, 1),
        "runs": runs,
        "headline_beam": headline["num_beams"],
        "headline_mean_s": round(headline["mean_s"], 2),
        "space_budget_s": 10.0,
        "verdict": _verdict(headline["mean_s"]),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))

    print(f"model: {MODEL}   threads: {args.threads}   peak RSS: {peak_rss:.0f} MB")
    for r in runs:
        print(
            f"  beams={r['num_beams']}  mean={r['mean_s']:.2f}s  p95={r['p95_s']:.2f}s  "
            f"min={r['min_s']:.2f}s  max={r['max_s']:.2f}s  (n={r['n']})"
        )
    print(f'  sample: "{runs[-1]["sample_caption"]}"')
    print(f"verdict (beams={headline['num_beams']}, {headline['mean_s']:.2f}s vs 10s budget): {report['verdict']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
