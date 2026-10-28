#!/bin/bash
#SBATCH --job-name=mct-train
#SBATCH --account=aibt
#SBATCH --qos=aibt
#SBATCH --partition=aibt-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/cv/scratch/u/andera29/mct_pretraining/logs/train-%x-%j.out
#SBATCH --error=/cv/scratch/u/andera29/mct_pretraining/logs/train-%x-%j.err

# Args (passed via --export or sbatch --export=ALL,TRAIN_FRACTION=...,EXP_NAME=...)
: "${TRAIN_FRACTION:?TRAIN_FRACTION must be set (0.10 / 0.25 / 0.50 / 1.00)}"
: "${EXP_NAME:?EXP_NAME must be set}"
: "${SPLIT_FILE:=/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json}"
: "${INPUT_DIR:=/cv/scratch/u/andera29/mct_pretraining/cellarr}"
: "${OUTPUT_DIR:=/cv/scratch/u/andera29/mct_pretraining/runs}"
: "${WANDB_API_KEY:=}"
# Per-fraction overrides: submit_ablation.sh sets these from an EPOCHS target
# so training length + cosine T_max + warmup scale with training-data size.
: "${MAX_STEPS:=110000}"
: "${NUM_WARMUP_STEPS:=20000}"
: "${CKPT_PATH:=}"

set -euo pipefail

REPO=/cv/data/braid/andera29/projs/mct/mct_pub

# Activate micromamba env
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mct

cd "$REPO"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Train fraction: $TRAIN_FRACTION"
echo "Experiment: $EXP_NAME"

# NCCL / dataloader tuning
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

# Config mirrors /cv/scratch/u/andera29/mct_ckpts/12k_no_pseudobulk.ckpt
# (~100M params, 12 layers, d=512, hidden=768, seq=12k, bag=1024, lr=1.5e-4,
# warmup=20k). Trainer stops at 110k steps to match the reference ckpt.
# Note: repo scheduler is linear-warmup-then-flat; ckpt LR trajectory suggests
# original training used a scheduler not in this repo.
python cell_cell_interactions/modeling/train_cellarr.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXP_NAME" \
    --split-file "$SPLIT_FILE" \
    --train-fraction "$TRAIN_FRACTION" \
    --sample-key sample \
    --cell-type-key cell_type \
    --cell-type-mapping-file-path "$INPUT_DIR/cell_type_map.json" \
    --num-workers "${NUM_WORKERS:-8}" \
    --prefetch-factor 4 \
    --batch-size 4 \
    --accumulate-grad-batches 4 \
    --n-cells-bag 1024 \
    --max-seq-len 12000 \
    --min-n-cells-per-sample 256 \
    --min-cell-type-size 0.01 \
    --perc-min-expressed-cells-per-gene 0.01 \
    --max-steps "$MAX_STEPS" \
    --num-warmup-steps "$NUM_WARMUP_STEPS" \
    --val-check-interval 10000 \
    --limit-val-batches 2000 \
    --log-every-n-steps 1000 \
    --patience 15 \
    --save-checkpoint-every-n-steps 10000 \
    --gene-emb-dim 512 \
    --hidden-dim 768 \
    --n-heads 8 \
    --n-layers 12 \
    --dropout 0.2 \
    --transformer-dropout 0.1 \
    --lr 0.00015 \
    --weight-decay 0.02 \
    --mask-prob 0.15 \
    --mask-tissue-prob 0.25 \
    --alpha-reconstruction-loss-weight 0.01 \
    --gamma-tissue-loss-weight 0.05 \
    --precision bf16-true \
    --devices 1 \
    --strategy auto \
    --accelerator auto \
    --wandb-api-key "$WANDB_API_KEY" \
    ${CKPT_PATH:+--ckpt-path "$CKPT_PATH"}

echo "End: $(date)"
