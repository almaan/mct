#!/bin/bash
#SBATCH --job-name=mct-preprocess
#SBATCH --account=aibt
#SBATCH --qos=aibt
#SBATCH --partition=himem
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/cv/scratch/u/andera29/mct_pretraining/logs/preprocess-%j.out
#SBATCH --error=/cv/scratch/u/andera29/mct_pretraining/logs/preprocess-%j.err

set -euo pipefail

REPO=/cv/data/braid/andera29/projs/mct/mct_pub
PROCESSED_DIR=/cv/scratch/u/andera29/mct_pretraining/processed
TEMP_DIR=/cv/scratch/u/andera29/mct_pretraining/raw

# Activate micromamba env
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mct

cd "$REPO"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"

python cell_cell_interactions/pp/cellarr_dataset/run_download_preprocess_datasets.py \
    --processed-dir "$PROCESSED_DIR" \
    --temporary-dir "$TEMP_DIR"

echo "End: $(date)"
