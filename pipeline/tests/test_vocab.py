"""Stage 1.1 — vocabulary.

Pure tests on a hand-built vocab always run; the size-band guard on the built
data/vocab.json skips when it hasn't been generated.
"""

import pytest

from capit.config import config
from capit.data.records import train_captions
from capit.data.vocab import END, PAD, SPECIALS, START, UNK, Vocab

# a:5, b:4, c:5, d:1  → at min_freq=5 keep {a, c}; b and d drop.
CAPS = [["a", "c"]] * 5 + [["b"]] * 4 + [["d"]]


def test_specials_are_stable():
    v = Vocab.build(CAPS, min_freq=5)
    assert v.encode([PAD, START, END, UNK]) == [0, 1, 2, 3]


def test_min_freq_boundary():
    v = Vocab.build(CAPS, min_freq=5)
    assert v.encode(["a"]) != [v.word2id[UNK]]
    assert v.encode(["c"]) != [v.word2id[UNK]]
    assert v.encode(["b"]) == [v.word2id[UNK]]
    assert v.encode(["d"]) == [v.word2id[UNK]]


def test_min_freq_inclusive():
    v = Vocab.build(CAPS, min_freq=4)
    assert v.encode(["b"]) != [v.word2id[UNK]]


def test_unknown_maps_to_unk():
    v = Vocab.build(CAPS, min_freq=5)
    assert v.encode(["zzz"]) == [3]
    assert v.decode([3]) == [UNK]


def test_roundtrip_in_vocab():
    v = Vocab.build(CAPS, min_freq=5)
    assert v.decode(v.encode(["a", "c", "a"])) == ["a", "c", "a"]


def test_frequency_then_alpha_order():
    # a and c both count 5; alphabetical tiebreak puts a before c, both after specials.
    v = Vocab.build(CAPS, min_freq=5)
    assert v.encode(["a"]) == [4]
    assert v.encode(["c"]) == [5]


def test_distinct_count_ordering():
    caps = [["hi"]] * 7 + [["mid"]] * 6 + [["lo"]] * 5
    v = Vocab.build(caps, min_freq=5)
    assert v.encode(["hi", "mid", "lo"]) == [4, 5, 6]


def test_len():
    v = Vocab.build(CAPS, min_freq=5)
    assert len(v) == 6


def test_sha256_depends_only_on_content():
    v = Vocab.build(CAPS, min_freq=5)
    assert v.sha256() == Vocab(list(v.id2word)).sha256()
    assert v.sha256() != Vocab(SPECIALS + ["a", "c", "extra"]).sha256()


def test_decode_rejects_out_of_range():
    v = Vocab.build(CAPS, min_freq=5)
    with pytest.raises(ValueError):
        v.decode([len(v)])
    with pytest.raises(ValueError):
        v.decode([-1])


def test_init_rejects_bad_specials():
    with pytest.raises(ValueError):
        Vocab(["a", "b", "c", "d"])


def test_init_rejects_duplicates():
    with pytest.raises(ValueError):
        Vocab(SPECIALS + ["a", "a"])


def test_train_captions_excludes_val_test():
    records = [
        {"split": "train", "sentences": [{"tokens": ["traintoken"]}]},
        {"split": "val", "sentences": [{"tokens": ["valtoken"]}]},
        {"split": "test", "sentences": [{"tokens": ["testtoken"]}]},
    ]
    assert train_captions(records) == [["traintoken"]]


def test_train_captions_raises_when_no_train():
    with pytest.raises(ValueError):
        train_captions([{"split": "val", "sentences": [{"tokens": ["x"]}]}])


def test_save_load_identity(tmp_path):
    v = Vocab.build(CAPS, min_freq=5)
    path = tmp_path / "vocab.json"
    v.save(path)
    loaded = Vocab.load(path)
    assert loaded.id2word == v.id2word
    assert loaded.encode(["a", "zzz", "c"]) == v.encode(["a", "zzz", "c"])


@pytest.mark.skipif(
    not config.vocab_path.is_file(),
    reason="vocab not built (run: python pipeline/scripts/build_vocab.py)",
)
def test_vocab_size_in_band():
    v = Vocab.load(config.vocab_path)
    assert 2500 <= len(v) <= 3500, f"vocab size {len(v)} out of band — check min_freq filtering"
