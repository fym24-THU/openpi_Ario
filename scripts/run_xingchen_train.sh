#!/bin/bash
set -e

cd /home/fanyiming/openpi_Ario
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"
export WANDB_MODE=disabled
# Auto-detect network interface for NCCL (don't hardcode eth0)
NCCL_IF=$(cat /sys/class/net/*/operstate 2>/dev/null | grep -l up /sys/class/net/*/operstate 2>/dev/null | head -1 | cut -d'/' -f5 || echo "")
if [ -n "$NCCL_IF" ]; then
    export NCCL_SOCKET_IFNAME=$NCCL_IF
    export GLOO_SOCKET_IFNAME=$NCCL_IF
fi
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error.json

# Install transformers_replace patches
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# Number of GPUs (auto-detect or override via env)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi
echo "Using $NUM_GPUS GPU(s)"

echo "=== Step 0: Single Vision ==="
CONFIG_NAME="pi05_xingchen_bench_xc03_pp3_pp10"
echo "=== Step 1: Convert JAX weights to PyTorch (if not already done) ==="
PYTORCH_WEIGHT_DIR="./checkpoints/pi05_base_pytorch"
if [ -f "$PYTORCH_WEIGHT_DIR/model.safetensors" ]; then
    echo "PyTorch weights already exist at $PYTORCH_WEIGHT_DIR, skipping conversion."
else
    echo "Converting JAX weights to PyTorch format..."
    python examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
        --config_name "$CONFIG_NAME" \
        --output_path "$PYTORCH_WEIGHT_DIR" \
        --precision bfloat16
fi

echo "=== Step 2: Compute norm stats (if not already done) ==="
NORM_STATS_PATH=$(python -c "
from openpi.training.config import get_config
cfg = get_config('$CONFIG_NAME')
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
print(cfg.assets_dirs / data_cfg.repo_id / 'norm_stats.json')
")
echo "Norm stats path: $NORM_STATS_PATH"
if [ -f "$NORM_STATS_PATH" ]; then
    echo "Norm stats already exist at $NORM_STATS_PATH, skipping."
else
    echo "Computing normalization statistics for new data..."
    python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

echo "=== Step 3: Training (PyTorch DDP) ==="
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((RANDOM % 10000 + 20000))
echo "Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    scripts/train_pytorch.py "$CONFIG_NAME" --overwrite

echo "=== Done ==="
