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


echo "=== Step 1: Convert JAX weights to PyTorch (if not already done) ==="
PYTORCH_WEIGHT_DIR="./checkpoints/pi05_base_pytorch"
if [ -f "$PYTORCH_WEIGHT_DIR/model.safetensors" ]; then
    echo "PyTorch weights already exist at $PYTORCH_WEIGHT_DIR, skipping conversion."
else
    echo "Converting JAX weights to PyTorch format..."
    python examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
        --config_name pi05_xingchen_ario \
        --output_path "$PYTORCH_WEIGHT_DIR" \
        --precision bfloat16
fi

echo "=== Step 2: Compute norm stats (if not already done) ==="
NORM_STATS_PATH="./assets/pi05_xingchen_ario/xingchen/new_blocks/norm_stats.json"
if [ -f "$NORM_STATS_PATH" ]; then
    echo "Norm stats already exist at $NORM_STATS_PATH, skipping."
else
    echo "Computing normalization statistics for new data..."
    python scripts/compute_norm_stats.py --config-name pi05_xingchen_ario
fi

echo "=== Step 3: Training (PyTorch DDP) ==="
# Probe the first episode on OSS to verify all required camera videos.
echo "--- Multi-view training status ---"
python -c "
import boto3, botocore
from openpi.training.config import _CONFIGS_DICT

cfg = _CONFIGS_DICT['pi05_xingchen_ario']
if not cfg.data.multi_view:
    print('Multi-view (3-view) training: DISABLED (single-view: video.mp4)')
else:
    # Parse first s3 prefix to probe
    raw = cfg.data.s3_prefixes.strip().rstrip(',')
    first_prefix = raw.split(',')[0].strip()
    # Parse bucket and key prefix
    parts = first_prefix.split('://', 1)[1].split('/', 1)
    bucket, base_prefix = parts[0], parts[1] if len(parts) > 1 else ''

    s3 = boto3.client('s3',
        endpoint_url=cfg.data.s3_endpoint,
        config=botocore.config.Config(signature_version='s3v4'))

    # Find first episode subfolder
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
    episode_prefixes = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
    if not episode_prefixes:
        raise SystemExit('ERROR: no episodes found under ' + base_prefix)
    else:
        ep = episode_prefixes[0]
        cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
        missing = []
        for camera in cameras:
            key = ep + f'raw_video/{camera}.mp4'
            try:
                s3.head_object(Bucket=bucket, Key=key)
            except s3.exceptions.ClientError:
                missing.append(camera)
        if missing:
            raise SystemExit(f'ERROR: first episode {ep} is missing camera videos: {missing}')
        print('Multi-view (3-view) training: ENABLED')
        print('  Camera views verified:', ', '.join(cameras))
        print('  First episode:', ep)
"
echo "---"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((RANDOM % 10000 + 20000))
echo "Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    scripts/train_pytorch.py pi05_xingchen_ario --overwrite

echo "=== Done ==="
