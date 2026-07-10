#!/bin/bash
set -e

cd /home/fanyiming/openpi
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"
export WANDB_MODE=disabled
export https_proxy=192.168.48.27:18000
export http_proxy=192.168.48.27:18000

echo "=== Step 1: Computing norm stats (small dataset, max 512 frames) ==="
NORM_STATS_DIR="./assets/pi05_xingchen_fold_ario_debug/xingchen/fold_clothes"
if [ -d "$NORM_STATS_DIR" ] && [ "$(ls -A $NORM_STATS_DIR 2>/dev/null)" ]; then
    echo "Norm stats already exist at $NORM_STATS_DIR, skipping computation."
else
    python scripts/compute_norm_stats.py --config-name pi05_xingchen_fold_ario_debug --max-frames 512
fi

echo "=== Step 2: Training (500 steps, batch_size=8) ==="
python scripts/train.py pi05_xingchen_fold_ario_debug --overwrite

echo "=== Done ==="
