#!/bin/bash
set -e

cd /home/fanyiming/openpi_Ario
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"
export WANDB_MODE=disabled

echo "=== Training (JAX, debug config) ==="
python scripts/train.py pi05_xingchen_fold_ario_debug --overwrite

echo "=== Done ==="
