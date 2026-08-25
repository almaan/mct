#!/bin/bash
#SBATCH --job-name=mct-build-cellarr
#SBATCH --account=aibt
#SBATCH --qos=aibt
#SBATCH --partition=himem
#SBATCH --cpus-per-task=4
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/cv/scratch/u/andera29/mct_pretraining/logs/build-%j.out
#SBATCH --error=/cv/scratch/u/andera29/mct_pretraining/logs/build-%j.err

set -euo pipefail

REPO=/cv/data/braid/andera29/projs/mct/mct_pub
PROCESSED_DIR=/cv/scratch/u/andera29/mct_pretraining/processed
OUTPUT_DIR=/cv/scratch/u/andera29/mct_pretraining/cellarr

# Activate micromamba env
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mct

cd "$REPO"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"

python cell_cell_interactions/pp/cellarr_dataset/run_build_cellarr_dataset.py \
    --manifest "$PROCESSED_DIR/pretraining_manifest.json" \
    --output-dir "$OUTPUT_DIR" \
    --num-threads 1

echo "End: $(date)"
