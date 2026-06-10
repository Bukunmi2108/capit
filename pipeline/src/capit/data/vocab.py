"""Vocabulary: a word<->id mapping with frequency-filtered tokens."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]


class Vocab:
    def __init__(self, words: list[str]) -> None:
        if list(words[:4]) != SPECIALS:
            raise ValueError(f"ids 0-3 must be {SPECIALS}, got {list(words[:4])}")
        if len(set(words)) != len(words):
            raise ValueError("vocab contains duplicate words")
        self.id2word = list(words)
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

    @classmethod
    def build(cls, captions: list[list[str]], min_freq: int = 5) -> "Vocab":
        counts: Counter[str] = Counter()
        for tokens in captions:
            counts.update(tokens)
        kept = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls(SPECIALS + kept)

    def encode(self, words: list[str]) -> list[int]:
        unk = self.word2id[UNK]
        return [self.word2id.get(w, unk) for w in words]

    def decode(self, ids: list[int]) -> list[str]:
        n = len(self.id2word)
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(f"decode: id {i} out of range [0, {n})")
        return [self.id2word[i] for i in ids]

    def __len__(self) -> int:
        return len(self.id2word)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.id2word, ensure_ascii=False).encode()).hexdigest()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.id2word, ensure_ascii=False, indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | str) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))
