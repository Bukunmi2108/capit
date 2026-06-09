"""CaptionDataset.

Runs against the dev subsample + built vocab; skips when either is absent.
"""

from collections import Counter
import pytest
import torch
from torch.utils.data import DataLoader

from capit.config import config
from capit.data.dataset import CaptionDataset, build_transform, collate_fn
from capit.data.vocab import END, PAD, START, Vocab

pytestmark = pytest.mark.skipif(
    not (config.subsample_json.is_file() and config.vocab_path.is_file()),
    reason="dev subsample or vocab not built",
)


@pytest.fixture(scope="module")
def vocab() -> Vocab:
    return Vocab.load(config.vocab_path)


@pytest.fixture(scope="module")
def train_ds(vocab: Vocab) -> CaptionDataset:
    return CaptionDataset(config.subsample_root, "train", vocab, build_transform())


def test_len_is_five_per_image(train_ds: CaptionDataset):
    n_images = len({r["filename"] for r, _ in train_ds.pairs})
    assert len(train_ds) == 5 * n_images
    assert len(train_ds) == 5 * config.subsample_train


def test_each_image_has_exactly_five_pairs(train_ds: CaptionDataset):
    per_image = Counter(r["filename"] for r, _ in train_ds.pairs)
    assert set(per_image.values()) == {5}


def test_item_shapes_and_dtypes(train_ds: CaptionDataset):
    img, ids, clen = train_ds[0]
    assert img.shape == (3, config.crop, config.crop)
    assert img.dtype == torch.float32
    assert ids.shape == (train_ds.max_len,)
    assert ids.dtype == torch.long
    assert isinstance(clen, int)


def test_start_end_and_padding(train_ds: CaptionDataset, vocab: Vocab):
    _, ids, clen = train_ds[0]
    assert ids[0].item() == vocab.word2id[START]
    assert ids[clen - 1].item() == vocab.word2id[END]
    assert all(ids[j].item() == vocab.word2id[PAD] for j in range(clen, train_ds.max_len))


def test_references_five_per_image(vocab: Vocab):
    ds = CaptionDataset(config.subsample_root, "test", vocab, build_transform())
    refs = ds.references()
    assert len(refs) == config.subsample_test
    assert all(len(r) == 5 for r in refs.values())


def test_references_are_raw_str_tokens(vocab: Vocab):
    ds = CaptionDataset(config.subsample_root, "test", vocab, build_transform())
    refs = ds.references()
    assert all(isinstance(tok, str) for caps in refs.values() for caption in caps for tok in caption)


def test_iter_images_once_per_image(vocab: Vocab):
    ds = CaptionDataset(config.subsample_root, "test", vocab, build_transform())
    ids = [iid for iid, _ in ds.iter_images()]
    assert len(ids) == config.subsample_test
    assert len(set(ids)) == config.subsample_test


def test_image_id_consistent_across_eval_views(vocab: Vocab):
    ds = CaptionDataset(config.subsample_root, "test", vocab, build_transform())
    ref_ids = set(ds.references().keys())
    img_ids = {iid for iid, _ in ds.iter_images()}
    assert img_ids == ref_ids == set(range(config.subsample_test))


def test_unknown_split_raises(vocab: Vocab):
    with pytest.raises(ValueError):
        CaptionDataset(config.subsample_root, "nope", vocab, build_transform())


def test_collate_sorts_descending_with_alignment(train_ds: CaptionDataset, vocab: Vocab):
    items = sorted((train_ds[i] for i in range(8)), key=lambda x: x[2])  # ascending → force re-sort
    images, captions, lengths = collate_fn(items)
    lens = lengths.tolist()
    assert images.shape == (8, 3, config.crop, config.crop)
    assert captions.shape == (8, train_ds.max_len)
    assert lens == sorted(lens, reverse=True)
    assert all(n <= train_ds.max_len for n in lens)
    pad = vocab.word2id[PAD]
    for row, n in enumerate(lens):
        assert int((captions[row] != pad).sum()) == n


def test_determinism_seeded(train_ds: CaptionDataset):
    def first_batch() -> torch.Tensor:
        g = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g, collate_fn=collate_fn)
        return next(iter(loader))[1]

    assert torch.equal(first_batch(), first_batch())
