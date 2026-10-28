#!/bin/bash
#SBATCH --job-name=mct-smoke
#SBATCH --account=aibt
#SBATCH --qos=aibt
#SBATCH --partition=aibt-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/cv/scratch/u/andera29/mct_pretraining/logs/smoke-%j.out
#SBATCH --error=/cv/scratch/u/andera29/mct_pretraining/logs/smoke-%j.err

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
echo "GPUs: $(nvidia-smi -L)"

python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('arch list:', torch.cuda.get_arch_list())
print('sm_100 supported:', 'sm_100' in torch.cuda.get_arch_list())
print('device:', torch.cuda.get_device_name(0))
"

python scripts/smoke_test_ablation.py

echo "End: $(date)"
