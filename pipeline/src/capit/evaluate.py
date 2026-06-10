"""Official BLEU/CIDEr on the test split.

Generates one caption per test image (beam search) and scores it against the 5 references with the pycocoevalcap Python scorers used directly on pre-tokenized strings. Encoder features are cached once and reused across beam widths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from capit.checkpoint import load
from capit.config import config
from capit.data.dataset import CaptionDataset, build_transform
from capit.data.vocab import END, START, Vocab
from capit.decode import beam_search
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder

METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"]


def score(
    references: dict[int, list[list[str]]], candidates: dict[int, list[str]]
) -> dict[str, float]:
    gts = {i: [" ".join(toks) for toks in refs] for i, refs in references.items()}
    res = {i: [" ".join(candidates[i])] for i in references}
    bleu = Bleu(4).compute_score(gts, res, verbose=0)[0]
    cider = Cider().compute_score(gts, res)[0]
    return {
        "BLEU-1": float(bleu[0]) * 100,
        "BLEU-2": float(bleu[1]) * 100,
        "BLEU-3": float(bleu[2]) * 100,
        "BLEU-4": float(bleu[3]) * 100,
        "CIDEr": float(cider) * 100,
    }


@torch.no_grad()
def evaluate(
    data_root, ckpt_path, beams: tuple[int, ...] = (1, 3, 5)
) -> dict[int, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(config.vocab_path)
    encoder = Encoder(pretrained=True).to(device).eval()
    decoder = Decoder(vocab_size=len(vocab)).to(device)
    state = load(ckpt_path)
    if state.vocab_sha256 != vocab.sha256():
        raise ValueError(f"vocab mismatch: ckpt {state.vocab_sha256[:8]} != vocab {vocab.sha256()[:8]}")
    decoder.load_state_dict(state.model_state)

    ds = CaptionDataset(data_root, "test", vocab, build_transform())
    references = ds.references()
    features = {iid: encoder(img.unsqueeze(0).to(device)) for iid, img in ds.iter_images()}

    sid, eid = vocab.word2id[START], vocab.word2id[END]
    results: dict[int, dict[str, float]] = {}
    for k in beams:
        candidates = {
            iid: vocab.decode(beam_search(decoder, f, sid, eid, k=k)[0])
            for iid, f in features.items()
        }
        results[k] = score(references, candidates)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(config.flickr8k_dir))
    parser.add_argument("--ckpt", default=str(config.ckpt_dir / "best.pt"))
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out-json", help="write the results table to this path for downstream use")
    args = parser.parse_args()

    results = evaluate(args.data_root, args.ckpt, tuple(args.beams))
    header = "beam  " + "  ".join(f"{m:>7}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for k, s in results.items():
        print(f"{k:>4}  " + "  ".join(f"{s[m]:>7.2f}" for m in METRICS))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
