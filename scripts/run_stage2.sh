#!/usr/bin/env bash
# Train a downstream detector with GRSR (500 epochs x 3 refinement rounds) and evaluate it.
# Usage: bash scripts/run_stage2.sh <DATA_ROOT> [NETWORK=dnanet|acmnet] [GPU=0] [SEED=123]
set -euo pipefail
DATA_ROOT=${1:?usage: run_stage2.sh <DATA_ROOT> [NETWORK] [GPU] [SEED]}
NETWORK=${2:-dnanet}
GPU=${3:-0}
SEED=${4:-123}
NAME=$(basename "$DATA_ROOT")

python train.py \
    --data-root "$DATA_ROOT" \
    --network "$NETWORK" \
    --loss grsr \
    --update_mode union \
    --epochs 500 --iterations 3 --batch-size 4 \
    --gpus "$GPU" --seed "$SEED" \
    --save-tag "grsr_${NETWORK}_${NAME}_s${SEED}"
