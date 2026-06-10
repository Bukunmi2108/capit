"""Stage 6.1 — backend API. Needs the local artifact + cached BLIP; skips cleanly otherwise."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ARTIFACT_DIR, app

from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "capit-sat.pt").exists(),
    reason="no local serving artifact; run scripts/export_artifact.py first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png(size=(64, 48), mode="RGB", exif=None) -> bytes:
    img = Image.new(mode, size, "red")
    buf = io.BytesIO()
    if exif is not None:
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_caption_happy_sat(client):
    r = client.post("/caption?model=sat&beam=3", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    sat = r.json()["sat"]
    assert sat["caption"] and len(sat["words"]) == len(sat["attention"])
    assert all(len(row) == 196 for row in sat["attention"])
    assert set(sat["crop"]) == {"x", "y", "w", "h"}
    assert sat["beams"][0]["caption"]


def test_caption_both_has_blip(client):
    r = client.post("/caption?model=both", files={"file": ("x.png", _png(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["blip"]["caption"] and body["sat"]["caption"]


def test_junk_bytes_is_422(client):
    r = client.post("/caption?model=sat", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 422


def test_oversized_is_413(client):
    big = b"\xff" * (8 * 1024 * 1024 + 1)
    r = client.post("/caption?model=sat", files={"file": ("x.png", big, "image/png")})
    assert r.status_code == 413


def test_rgba_accepted(client):
    r = client.post("/caption?model=sat", files={"file": ("x.png", _png(mode="RGBA"), "image/png")})
    assert r.status_code == 200


def test_exif_orientation_applied(client):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW: a saved 400x200 displays upright as 200x400 (portrait)
    r = client.post("/caption?model=sat", files={"file": ("x.jpg", _png((400, 200), exif=exif), "image/jpeg")})
    crop = r.json()["sat"]["crop"]
    assert crop["w"] > crop["h"]  # portrait-upright crops wider in fractions; ignoring EXIF would flip this
