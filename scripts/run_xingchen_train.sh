#!/bin/bash
set -e

cd /home/fanyiming/openpi
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"

echo "=== Step 1: Computing norm stats ==="
python scripts/compute_norm_stats.py --config-name pi05_xingchen_fold_ario

echo "=== Step 2: Training ==="
python scripts/train.py pi05_xingchen_fold_ario

echo "=== Done ==="
