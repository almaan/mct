#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODEOCEAN_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${CODEOCEAN_DATA_DIR:-/data}"
RESULTS_DIR="${CODEOCEAN_RESULTS_DIR:-/results}"
STRICT_INPUTS="${MCT_STRICT_INPUTS:-1}"
CREATE_CODEOCEAN_SYMLINKS="${MCT_CREATE_CODEOCEAN_SYMLINKS:-}"

if [ -z "${CREATE_CODEOCEAN_SYMLINKS}" ]; then
  if [ "${CODE_DIR}" = "/code" ]; then
    CREATE_CODEOCEAN_SYMLINKS="1"
  else
    CREATE_CODEOCEAN_SYMLINKS="0"
  fi
fi

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/tables" "${RESULTS_DIR}/figures"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="${RESULTS_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

if [ -d "${DATA_ROOT}/tutorials/data" ]; then
  TUTORIAL_DATA_DIR="${DATA_ROOT}/tutorials/data"
elif [ -d "${CODE_DIR}/tutorials/data" ]; then
  TUTORIAL_DATA_DIR="${CODE_DIR}/tutorials/data"
else
  TUTORIAL_DATA_DIR="${DATA_ROOT}/tutorials/data"
fi

if [ -n "${CODEOCEAN_CHECKPOINT_PATH:-}" ]; then
  CHECKPOINT_PATH="${CODEOCEAN_CHECKPOINT_PATH}"
elif [ -f "${DATA_ROOT}/checkpoints/multicellular_transformer.ckpt" ]; then
  CHECKPOINT_PATH="${DATA_ROOT}/checkpoints/multicellular_transformer.ckpt"
elif [ -f "${CODE_DIR}/checkpoints/multicellular_transformer.ckpt" ]; then
  CHECKPOINT_PATH="${CODE_DIR}/checkpoints/multicellular_transformer.ckpt"
else
  CHECKPOINT_PATH="${DATA_ROOT}/checkpoints/multicellular_transformer.ckpt"
fi

if [ "${CREATE_CODEOCEAN_SYMLINKS}" = "1" ] && [ -d "${DATA_ROOT}/tutorials/data" ] && [ ! -e "${CODE_DIR}/tutorials/data" ]; then
  mkdir -p "${CODE_DIR}/tutorials"
  ln -s "${DATA_ROOT}/tutorials/data" "${CODE_DIR}/tutorials/data"
fi

if [ "${CREATE_CODEOCEAN_SYMLINKS}" = "1" ] && [ -d "${DATA_ROOT}/checkpoints" ] && [ ! -e "${CODE_DIR}/checkpoints" ]; then
  ln -s "${DATA_ROOT}/checkpoints" "${CODE_DIR}/checkpoints"
fi

{
  echo "Code directory: ${CODE_DIR}"
  echo "Data root: ${DATA_ROOT}"
  echo "Tutorial data: ${TUTORIAL_DATA_DIR}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Results directory: ${RESULTS_DIR}"
  echo "Strict inputs: ${STRICT_INPUTS}"
  echo "Create CodeOcean symlinks: ${CREATE_CODEOCEAN_SYMLINKS}"
} | tee "${RESULTS_DIR}/logs/run_configuration.log"

python -m compileall -q "${CODE_DIR}/cell_cell_interactions"
python -m pytest -q "${CODE_DIR}/tests" | tee "${RESULTS_DIR}/logs/pytest.log"

SCRIPT_ARGS=(
  --tutorial-data-dir "${TUTORIAL_DATA_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --results-dir "${RESULTS_DIR}"
  --code-dir "${CODE_DIR}"
)

if [ "${STRICT_INPUTS}" = "0" ]; then
  SCRIPT_ARGS+=(--allow-missing-checkpoint)
fi

python -u "${CODE_DIR}/codeocean/reproduce_reviewer_outputs.py" "${SCRIPT_ARGS[@]}" \
  | tee "${RESULTS_DIR}/logs/reproduce_reviewer_outputs.log"
