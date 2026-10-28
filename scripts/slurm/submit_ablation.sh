#!/bin/bash
# Submit ablation as independent SLURM jobs.
# Each job trains on a nested subset of the shared split_file.
#
# Behavior modes:
#   * Default (or EPOCHS unset): fixed max_steps=110000 across all fractions
#     (same as before — compute-matched panel).
#   * EPOCHS=<N>: max_steps and num_warmup_steps scale with each fraction's
#     batches-per-epoch so every sample is visited ~N times, matching a
#     data-scaling study convention.
#
# Env vars:
#   EPOCHS         optional; if set, per-fraction max_steps = ceil(N * batches_per_epoch / accum)
#   FRACTIONS      space-separated fractions to submit (default: "0.10 0.25 0.50 1.00")
#   EXP_TAG        experiment name prefix (default: "paperhp")
#   ACCUM          accumulate_grad_batches (default: 4, must match sbatch)
#
# Usage:
#   bash submit_ablation.sh [SPLIT_FILE] [OUTPUT_DIR]
#   EPOCHS=3 FRACTIONS="0.10 0.25 0.50" EXP_TAG=epochs3 bash submit_ablation.sh

set -euo pipefail

REPO=/cv/data/braid/andera29/projs/mct/mct_pub
SPLIT_FILE="${1:-/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json}"
OUTPUT_DIR="${2:-/cv/scratch/u/andera29/mct_pretraining/runs}"

if [[ ! -f "$SPLIT_FILE" ]]; then
    echo "Split file not found: $SPLIT_FILE" >&2
    echo "Generate it with make_ablation_split.py first." >&2
    exit 1
fi

mkdir -p /cv/scratch/u/andera29/mct_pretraining/logs
mkdir -p "$OUTPUT_DIR"

EXP_TAG="${EXP_TAG:-paperhp}"
FRACTIONS="${FRACTIONS:-0.10 0.25 0.50 1.00}"
ACCUM="${ACCUM:-4}"

# Batches per epoch = number of unique cells in the train_fraction subset,
# read from the CellArr build (already produced by make_ablation_split.py
# derived from post-filter counts). Table stays in sync with the actual
# 3474-sample corpus; if the corpus changes, regenerate this via a query.
declare -A BATCHES_PER_EPOCH=(
    ["0.05"]=15828
    ["0.10"]=29899
    ["0.25"]=84134
    ["0.50"]=186249
    ["1.00"]=351043
)

for FRAC in $FRACTIONS; do
    EXP_NAME="ablation_${EXP_TAG}_frac${FRAC}"
    EXPORT_KV="ALL,TRAIN_FRACTION=$FRAC,EXP_NAME=$EXP_NAME,SPLIT_FILE=$SPLIT_FILE,OUTPUT_DIR=$OUTPUT_DIR"

    if [[ -n "${EPOCHS:-}" ]]; then
        BPE=${BATCHES_PER_EPOCH[$FRAC]:-0}
        if [[ "$BPE" -eq 0 ]]; then
            echo "No batches_per_epoch entry for fraction $FRAC" >&2
            exit 1
        fi
        # max_steps = ceil(EPOCHS * BPE / ACCUM)
        MAX_STEPS=$(( (EPOCHS * BPE + ACCUM - 1) / ACCUM ))
        # warmup = 10% of max_steps, clamped to [2000, 20000]
        WARM=$(( MAX_STEPS / 10 ))
        if [[ "$WARM" -lt 2000 ]]; then WARM=2000; fi
        if [[ "$WARM" -gt 20000 ]]; then WARM=20000; fi
        EXPORT_KV="$EXPORT_KV,MAX_STEPS=$MAX_STEPS,NUM_WARMUP_STEPS=$WARM"
        echo "Submitting: $EXP_NAME  (E=$EPOCHS -> max_steps=$MAX_STEPS, warmup=$WARM)"
    else
        echo "Submitting: $EXP_NAME  (fixed max_steps default from sbatch)"
    fi

    sbatch \
        --job-name="mct-${EXP_TAG}-${FRAC}" \
        --export="$EXPORT_KV" \
        "$REPO/scripts/slurm/train_ablation.sbatch"
done

echo "Submitted. Watch: squeue -u $USER --account=aibt"
