#!/bin/bash
set -e

cd /home/fanyiming/openpi
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"

# Disable torch.compile (Triton ptxas doesn't support sm_103a on this cluster)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Evaluate checkpoint on training set
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/pi05_xingchen_fold_ario_debug/pi05_xingchen_fold_ario_debug/500}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

echo "=== Evaluating checkpoint: $CHECKPOINT_DIR ==="
echo "=== Num samples: $NUM_SAMPLES ==="

python scripts/eval_xingchen_train.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config_name pi05_xingchen_fold_ario_debug \
    --num_samples "$NUM_SAMPLES" \
    --device cuda

echo "=== Evaluation done ==="
