# capit backend — HF Spaces (Docker SDK). Build context = repo root:
#   docker build -f backend/Dockerfile -t capit-api .
FROM python:3.12-slim

# uv, pinned to what the lockfile was authored with
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

# HF Spaces require a non-root user with UID 1000; set it up before any COPY
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/home/user/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CAPIT_ARTIFACT_REPO=Bukunmi2108/capit-sat \
    PATH=/home/user/venv/bin:$PATH

WORKDIR /home/user/app

# the capit package (model classes) the backend imports, then the backend project
COPY --chown=user pipeline/ ./pipeline/
COPY --chown=user backend/ ./backend/

WORKDIR /home/user/app/backend
RUN uv sync --frozen --no-dev

# bake weights into image layers so the Space has no cold-start downloads
RUN uv run python -c "from huggingface_hub import hf_hub_download as d; d('Bukunmi2108/capit-sat','capit-sat.pt'); d('Bukunmi2108/capit-sat','vocab.json')" \
 && uv run python -c "from transformers import BlipForConditionalGeneration as M, BlipProcessor as P; m='Salesforce/blip-image-captioning-base'; P.from_pretrained(m); M.from_pretrained(m)"

# weights are baked above — serve from cache only, no runtime Hub calls (faster cold start)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
