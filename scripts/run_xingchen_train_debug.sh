#!/bin/bash
set -e

cd /home/fanyiming/openpi
export HOME=/home/fanyiming
source .venv/bin/activate

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"
export WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

# Number of GPUs (auto-detect or override via env)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi
echo "Using $NUM_GPUS GPU(s)"


echo "=== Step 1: Convert JAX weights to PyTorch (if not already done) ==="
PYTORCH_WEIGHT_DIR="./checkpoints/pi05_base_pytorch"
if [ -f "$PYTORCH_WEIGHT_DIR/model.safetensors" ]; then
    echo "PyTorch weights already exist at $PYTORCH_WEIGHT_DIR, skipping conversion."
else
    echo "Converting JAX weights to PyTorch format..."
    python examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
        --config_name pi05_xingchen_fold_ario_debug \
        --output_path "$PYTORCH_WEIGHT_DIR" \
        --precision bfloat16
fi

echo "=== Step 2: Training (PyTorch DDP, 500 steps, batch_size=8) ==="
MASTER_ADDR=$(hostname -I | awk '{print $1}')
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    --master_addr="$MASTER_ADDR" --master_port=29500 \
    scripts/train_pytorch.py pi05_xingchen_fold_ario_debug --overwrite

echo "=== Done ==="
