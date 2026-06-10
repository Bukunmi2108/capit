#!/usr/bin/env bash
# Assemble the HF Space (Docker SDK) staging tree and force-push it.
#   ./scripts/sync_space.sh --dry-run   # assemble + inspect, no push
#   HF_TOKEN=hf_xxx ./scripts/sync_space.sh   # create-if-needed + deploy
# The Space is a build artifact — never edit it by hand; re-run this to update.
set -euo pipefail

HF_USER="${HF_USER:-Bukunmi2108}"
SPACE="${SPACE:-capit}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Dockerfile at the Space root (HF builds from there); same COPY paths as a local build.
cp "$REPO_ROOT/backend/Dockerfile" "$STAGE/Dockerfile"
cp "$REPO_ROOT/.dockerignore" "$STAGE/.dockerignore"

# code: pipeline/ (model classes) + backend/ (the app), excluding venvs/caches at copy time
rsync -a \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' \
  "$REPO_ROOT/pipeline" "$REPO_ROOT/backend" "$STAGE/"

# the YAML header is what makes this a Docker Space
cat > "$STAGE/README.md" <<'EOF'
---
title: capit
emoji: 🔎
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: A glass-box image captioner (Show, Attend and Tell) vs BLIP.
---

# capit — backend

Side-by-side captioning API: a from-scratch *Show, Attend and Tell* model (glass box —
per-word attention + rejected beams) and BLIP (closed box). Code:
https://github.com/Bukunmi2108/capit
EOF

echo "staged → $STAGE"
(cd "$STAGE" && find . -type f | sort | sed 's/^/  /')
du -sh "$STAGE" | awk '{print "context size:", $1}'

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] not pushing. run without --dry-run (HF_TOKEN set) to deploy."
  exit 0
fi

: "${HF_TOKEN:?set HF_TOKEN (a write token) to push}"
hf repo create "$HF_USER/$SPACE" --repo-type space --space-sdk docker || true

cd "$STAGE"
git init -q -b main
git add -A
git -c user.email="deploy@capit" -c user.name="capit-deploy" commit -qm "deploy capit backend"
git push -f "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE" main

echo "pushed → https://huggingface.co/spaces/$HF_USER/$SPACE"
echo "after the build: curl https://${HF_USER}-${SPACE}.hf.space/health"
