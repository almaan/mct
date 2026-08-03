#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-/tmp/multicellular-transformer-codeocean-capsule}"
CODE_OUT="${OUT_DIR}/code"
DATA_OUT="${OUT_DIR}/data"

if [ -e "${OUT_DIR}" ]; then
  echo "Refusing to overwrite existing staging directory: ${OUT_DIR}" >&2
  echo "Remove it or choose a different output directory." >&2
  exit 1
fi

mkdir -p "${CODE_OUT}" "${DATA_OUT}/tutorials" "${DATA_OUT}/checkpoints"

rsync -a \
  --exclude ".git/" \
  --exclude ".idea/" \
  --exclude ".vscode/" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ipynb_checkpoints/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude "data/" \
  --exclude "checkpoints/" \
  --exclude "logs/" \
  --exclude "wandb/" \
  --exclude "tutorials/data/" \
  --exclude "tutorials/old/" \
  "${REPO_DIR}/" "${CODE_OUT}/"

rsync -a \
  --exclude ".ipynb_checkpoints/" \
  --exclude ".DS_Store" \
  "${REPO_DIR}/tutorials/data/" "${DATA_OUT}/tutorials/data/"

if [ -f "${REPO_DIR}/checkpoints/multicellular_transformer.ckpt" ]; then
  cp "${REPO_DIR}/checkpoints/multicellular_transformer.ckpt" "${DATA_OUT}/checkpoints/multicellular_transformer.ckpt"
else
  cat > "${DATA_OUT}/checkpoints/README.txt" <<'EOF'
Upload the pretrained checkpoint here as:

multicellular_transformer.ckpt

The CodeOcean run expects it at:

/data/checkpoints/multicellular_transformer.ckpt
EOF
fi

cat <<EOF
Staged CodeOcean capsule files:

  Code: ${CODE_OUT}
  Data: ${DATA_OUT}

Upload the Code directory contents to /code and the Data directory contents to /data.
Set /code/run.sh as the CodeOcean file to run.
EOF
