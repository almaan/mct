#!/bin/bash
#SBATCH --job-name=mct-coexpr
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cv/scratch/u/andera29/mct_pretraining/logs/coexpr-%x-%j.out
#SBATCH --error=/cv/scratch/u/andera29/mct_pretraining/logs/coexpr-%x-%j.err

# Args (overridable via --export=ALL,VAR=...)
: "${PAIRS_CSV:=/cv/data/braid/andera29/projs/mct/mct_pub/rsc/rl_df_w_negs_with_gene_pairs.csv}"
: "${CELLARR_DIR:=/cv/scratch/u/andera29/mct_pretraining/cellarr}"
: "${SPLIT_FILE:=/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json}"
: "${TRAIN_FRACTION:=1.00}"
: "${CELLS_PER_SAMPLE:=128}"
: "${SAMPLING_SEED:=0}"
: "${OUTPUT_DIR:=/cv/data/braid/andera29/projs/mct/mct_pub/outputs/gene_pair_coexpression}"

set -euo pipefail

REPO=/cv/data/braid/andera29/projs/mct/mct_pub

export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mct

cd "$REPO"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "Train fraction: $TRAIN_FRACTION"
echo "Cells per sample: $CELLS_PER_SAMPLE"
echo "Output dir: $OUTPUT_DIR"

export TILEDB_DISABLE_S3=1
export TOKENIZERS_PARALLELISM=false

python scripts/eval_gene_pair_coexpression.py \
    --pairs-csv "$PAIRS_CSV" \
    --cellarr-dir "$CELLARR_DIR" \
    --split-file "$SPLIT_FILE" \
    --train-fraction "$TRAIN_FRACTION" \
    --cells-per-sample "$CELLS_PER_SAMPLE" \
    --sampling-seed "$SAMPLING_SEED" \
    --output-dir "$OUTPUT_DIR"

echo "End: $(date)"
