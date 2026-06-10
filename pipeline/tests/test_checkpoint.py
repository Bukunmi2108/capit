"""Stage 3.1 — checkpoint round-trip including optimizer state. Hermetic, no data."""

import torch
from torch import nn
from torch.optim import Adam

from capit.checkpoint import load, save


def _model() -> nn.Module:
    torch.manual_seed(13)
    return nn.Linear(4, 4)


def _step(model: nn.Module, opt: Adam) -> float:
    loss = model(torch.ones(2, 4)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def test_round_trip_identity(tmp_path):
    model = _model()
    opt = Adam(model.parameters())
    _step(model, opt)
    path = tmp_path / "ckpt.pt"
    save(path, model, opt, epoch=3, best_bleu4=12.5, best_epoch=2, vocab_sha256="abc")

    ts = load(path)
    assert (ts.epoch, ts.best_bleu4, ts.best_epoch, ts.vocab_sha256) == (3, 12.5, 2, "abc")
    fresh = nn.Linear(4, 4)
    fresh.load_state_dict(ts.model_state)
    for a, b in zip(fresh.state_dict().values(), model.state_dict().values()):
        assert torch.equal(a, b)


def test_optimizer_state_round_trips(tmp_path):
    model = _model()
    opt = Adam(model.parameters())
    _step(model, opt)
    _step(model, opt)
    path = tmp_path / "ckpt.pt"
    save(path, model, opt, epoch=2, best_bleu4=0.0, best_epoch=0, vocab_sha256="x")
    live = [_step(model, opt), _step(model, opt)]

    model2 = nn.Linear(4, 4)
    opt2 = Adam(model2.parameters())
    ts = load(path)
    model2.load_state_dict(ts.model_state)
    opt2.load_state_dict(ts.optim_state)
    resumed = [_step(model2, opt2), _step(model2, opt2)]

    assert live == resumed


def test_save_leaves_no_tmp(tmp_path):
    model = _model()
    opt = Adam(model.parameters())
    path = tmp_path / "ckpt.pt"
    save(path, model, opt, epoch=0, best_bleu4=0.0, best_epoch=0, vocab_sha256="x")
    assert path.exists()
    assert not (tmp_path / "ckpt.pt.tmp").exists()
