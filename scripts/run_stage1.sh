#!/usr/bin/env bash
# Generate the GRSR initial pseudo masks for one dataset.
# Usage: bash scripts/run_stage1.sh <DATA_ROOT>
set -euo pipefail
DATA_ROOT=${1:?usage: run_stage1.sh <DATA_ROOT>}

python generate_pseudo_labels.py --data-root "$DATA_ROOT"
