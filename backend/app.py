"""capit backend: side-by-side captions from the glass-box SAT model and BLIP.

Loads the Stage 4.4 serving artifact (local dir or HF Hub) + BLIP, exposes /health and
/caption. The SAT path returns per-word attention, the center-crop box, and the rejected beams.
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor

from capit.serving import caption as sat_caption
from capit.serving import center_crop_box, load_artifact, make_transform

MAX_BYTES = 8 * 1024 * 1024
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
ARTIFACT_REPO = os.environ.get("CAPIT_ARTIFACT_REPO")
ARTIFACT_DIR = Path(os.environ.get("CAPIT_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "data" / "artifact"))
ALLOWED_ORIGINS = os.environ.get("CAPIT_CORS", "http://localhost:5173,http://localhost:3000").split(",")

state: dict = {}


def _artifact_paths() -> tuple[Path, Path]:
    """Hub repo (Space; baked into the image cache at build) or a local dir (dev)."""
    if ARTIFACT_REPO:
        from huggingface_hub import hf_hub_download

        return (
            Path(hf_hub_download(ARTIFACT_REPO, "capit-sat.pt")),
            Path(hf_hub_download(ARTIFACT_REPO, "vocab.json")),
        )
    return ARTIFACT_DIR / "capit-sat.pt", ARTIFACT_DIR / "vocab.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(2)
    artifact, vocab_path = _artifact_paths()
    encoder, decoder, vocab, preprocess = load_artifact(artifact, vocab_path)
    state.update(
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        transform=make_transform(preprocess),
        preprocess=preprocess,
        blip_processor=BlipProcessor.from_pretrained(BLIP_MODEL),
        blip=BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).eval(),
    )
    yield
    state.clear()


app = FastAPI(title="Capit", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)


async def _ingest(file: UploadFile) -> Image.Image:
    body = await file.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_BYTES // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(body))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, f"could not decode image: {exc}") from exc
    return (ImageOps.exif_transpose(image) or image).convert("RGB")


@torch.no_grad()
def _sat(image: Image.Image, beam: int) -> dict:
    t0 = time.perf_counter()
    tensor = state["transform"](image)
    words, alphas, beams = sat_caption(state["encoder"], state["decoder"], state["vocab"], tensor, k=beam)
    decode = state["vocab"].decode
    return {
        "caption": " ".join(words),
        "words": words,
        "attention": [[round(a, 5) for a in row] for row in alphas.tolist()],
        "crop": center_crop_box(image.width, image.height, state["preprocess"]),
        "beams": [{"caption": " ".join(decode(toks)), "score": round(score, 3)} for toks, score in beams],
        "decode_ms": round((time.perf_counter() - t0) * 1000),
    }


@torch.no_grad()
def _blip(image: Image.Image, beam: int) -> dict:
    t0 = time.perf_counter()
    inputs = state["blip_processor"](image, return_tensors="pt")
    out = state["blip"].generate(**inputs, num_beams=beam, max_new_tokens=30)
    text = state["blip_processor"].decode(out[0], skip_special_tokens=True)
    return {"caption": text, "decode_ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Capit captioning API! Visit /docs for usage details."}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/caption")
async def caption_endpoint(
    file: UploadFile,
    model: str = Query("both", pattern="^(sat|blip|both)$"),
    beam: int = Query(3, ge=1, le=10),
) -> dict:
    image = await _ingest(file)
    result: dict = {}
    if model in ("sat", "both"):
        result["sat"] = _sat(image, beam)
    if model in ("blip", "both"):
        result["blip"] = _blip(image, beam)
    return result
