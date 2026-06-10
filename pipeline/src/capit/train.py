"""Train the capit model.

CLI: python -m capit.train --data-root <root> --ckpt-dir <dir> --resume auto

Checkpoints the decoder + optimizer every epoch (latest.pt) and on val-BLEU improvement (best.pt); the frozen encoder is reconstructed from ImageNet weights, not stored. Early stops on val BLEU-4 (nltk, a cheap relative signal — reported scores come from Stage 4.2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from capit.checkpoint import load, save
from capit.config import config
from capit.data.dataset import CaptionDataset, build_transform, collate_fn
from capit.data.vocab import END, START, Vocab
from capit.losses import caption_loss
from capit.models.decoder import Decoder
from capit.models.encoder import Encoder


def corpus_bleu4(references: dict[int, list[list[str]]], candidates: dict[int, list[str]]) -> float:
    order = list(references)
    if not order:
        raise ValueError("corpus_bleu4: no references (val split empty?)")
    refs = [references[i] for i in order]
    hyps = [candidates[i] for i in order]
    empty = sum(not h for h in hyps)
    if empty:
        print(f"warning: {empty}/{len(hyps)} val captions are empty (BLEU will read low)")
    return cast(float, corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)) * 100


def _train_one_epoch(encoder, decoder, loader, opt, device) -> float:
    decoder.train()
    total = 0.0
    for images, captions, lengths in loader:
        images, captions = images.to(device), captions.to(device)
        with torch.no_grad():
            features = encoder(images)
        logits, alphas, _ = decoder(features, captions, lengths)
        loss = caption_loss(logits, alphas, captions)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss.item()} — aborting before it corrupts latest.pt")
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(decoder.parameters(), config.grad_clip)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _evaluate(encoder, decoder, ds, vocab, device) -> tuple[float, dict[int, list[str]]]:
    decoder.eval()
    candidates: dict[int, list[str]] = {}
    for image_id, image in ds.iter_images():
        features = encoder(image.unsqueeze(0).to(device))
        ids = decoder.greedy(features, vocab.word2id[START], vocab.word2id[END])[0]
        candidates[image_id] = vocab.decode(ids)
    return corpus_bleu4(ds.references(), candidates), candidates


def train(
    data_root: str | Path,
    ckpt_dir: str | Path,
    resume: str = "auto",
    max_epochs: int | None = None,
    num_workers: int | None = None,
) -> None:
    max_epochs = config.max_epochs if max_epochs is None else max_epochs
    num_workers = config.num_workers if num_workers is None else num_workers
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    latest_path, best_path = ckpt_dir / "latest.pt", ckpt_dir / "best.pt"

    vocab = Vocab.load(config.vocab_path)
    transform = build_transform()
    train_ds = CaptionDataset(data_root, "train", vocab, transform)
    val_ds = CaptionDataset(data_root, "val", vocab, transform)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    encoder = Encoder(pretrained=True).to(device)
    decoder = Decoder(vocab_size=len(vocab)).to(device)
    opt = Adam(decoder.parameters(), lr=config.decoder_lr)

    start_epoch, best_bleu4, best_epoch = 0, -1.0, 0
    if resume == "auto" and latest_path.is_file():
        state = load(latest_path)
        if state.vocab_sha256 != vocab.sha256():
            raise ValueError("vocab mismatch: checkpoint vocab_sha256 != current vocab.json")
        decoder.load_state_dict(state.model_state)
        opt.load_state_dict(state.optim_state)
        start_epoch, best_bleu4, best_epoch = state.epoch + 1, state.best_bleu4, state.best_epoch
        print(f"resumed at epoch {start_epoch} (best_bleu4 {best_bleu4:.2f} at epoch {best_epoch})")

    for epoch in range(start_epoch, max_epochs):
        train_loss = _train_one_epoch(encoder, decoder, train_loader, opt, device)
        val_bleu4, candidates = _evaluate(encoder, decoder, val_ds, vocab, device)

        improved = val_bleu4 > best_bleu4
        if improved:
            best_bleu4, best_epoch = val_bleu4, epoch
        save(latest_path, decoder, opt, epoch, best_bleu4, best_epoch, vocab.sha256())
        if improved:
            save(best_path, decoder, opt, epoch, best_bleu4, best_epoch, vocab.sha256())

        sample = " ".join(next(iter(candidates.values())))
        flag = " *" if improved else ""
        print(f"epoch {epoch}  loss {train_loss:.4f}  val_bleu4 {val_bleu4:.2f}  best {best_bleu4:.2f}{flag}  | {sample}")
        if epoch - best_epoch >= config.patience:
            print(f"early stop: no val BLEU-4 improvement in {config.patience} epochs (last loss {train_loss:.4f})")
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(config.subsample_root))
    parser.add_argument("--ckpt-dir", default=str(config.ckpt_dir))
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    args = parser.parse_args()
    train(args.data_root, args.ckpt_dir, args.resume)


if __name__ == "__main__":
    main()
