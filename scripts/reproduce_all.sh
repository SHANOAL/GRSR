#!/usr/bin/env bash
# Run both stages on SIRST-v1, NUDT-SIRST and IRSTD-1K with DNANet and ACMNet.
# Usage: bash scripts/reproduce_all.sh [DATA_DIR=data] [GPU=0]
set -euo pipefail
DATA_DIR=${1:-data}
GPU=${2:-0}

for DATASET in SIRST-v1 NUDT-SIRST IRSTD-1K; do
    bash scripts/run_stage1.sh "$DATA_DIR/$DATASET"
    for NETWORK in dnanet acmnet; do
        bash scripts/run_stage2.sh "$DATA_DIR/$DATASET" "$NETWORK" "$GPU" 123
    done
done
