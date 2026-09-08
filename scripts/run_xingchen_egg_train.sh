#!/bin/bash
# Train pi05_xingchen_egg_ario and continuously deliver committed checkpoints.
#
# Required:
#   OSS_RUN_ROOT=s3+ali://bucket/path/to/checkpoints
#   WRITE__AWS_ACCESS_KEY_ID=...
#   WRITE__AWS_SECRET_ACCESS_KEY=...
#
# Optional:
#   EXP_NAME=egg_box TRAIN_MODE=overwrite|resume
set -euo pipefail

PROJECT_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ "${OPENPI_CONTAINER:-0}" != "1" ]]; then
    if [[ ! -f .venv/bin/activate ]]; then
        echo "Missing .venv. Run 'uv sync' first." >&2
        exit 1
    fi
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

CONFIG_NAME="pi05_xingchen_egg_ario"
EXP_NAME="${EXP_NAME:-egg_box}"
RUN_DIR="$PROJECT_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
NORM_STATS="$PROJECT_ROOT/assets/$CONFIG_NAME/xingchen/egg_box/norm_stats.json"

: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT to an s3+ali:// checkpoint destination}"
OSS_RUN_DIR="${OSS_RUN_ROOT%/}/$CONFIG_NAME/$EXP_NAME"

# Ario reads AWS-compatible names; the uploader uses ALI__-prefixed names.
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${ALIBABA_ACCESS_KEY_ID:-${WRITE__AWS_ACCESS_KEY_ID:-${ALI__AWS_ACCESS_KEY_ID:-}}}}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${ALIBABA_ACCESS_KEY_SECRET:-${WRITE__AWS_SECRET_ACCESS_KEY:-${ALI__AWS_SECRET_ACCESS_KEY:-}}}}"
export ALI__AWS_ACCESS_KEY_ID="${ALI__AWS_ACCESS_KEY_ID:-${WRITE__AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-${ALIBABA_ACCESS_KEY_ID:-}}}}"
export ALI__AWS_SECRET_ACCESS_KEY="${ALI__AWS_SECRET_ACCESS_KEY:-${WRITE__AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-${ALIBABA_ACCESS_KEY_SECRET:-}}}}"
export ALI__OSS_ENDPOINT="${ALI__OSS_ENDPOINT:-https://oss-cn-wulanchabu.aliyuncs.com}"
export ALI__AWS_S3_ADDRESSING_STYLE="${ALI__AWS_S3_ADDRESSING_STYLE:-virtual}"

: "${ALI__AWS_ACCESS_KEY_ID:?Set OSS access key credentials}"
: "${ALI__AWS_SECRET_ACCESS_KEY:?Set OSS secret key credentials}"

if [[ ! -f "$NORM_STATS" ]]; then
    echo "=== computing normalization statistics ==="
    python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

echo "=== local checkpoints: $RUN_DIR ==="
echo "=== OSS checkpoints:   $OSS_RUN_DIR ==="

UPLOADER_PID=""
cleanup() {
    if [[ -n "$UPLOADER_PID" ]]; then
        kill "$UPLOADER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python scripts/oss_ckpt_uploader.py \
    --local-run-dir "$RUN_DIR" \
    --oss-run-dir "$OSS_RUN_DIR" \
    --poll-secs 60 \
    --settle-secs 120 \
    --max-idle-polls 0 &
UPLOADER_PID=$!

TRAIN_MODE="${TRAIN_MODE:-overwrite}"
case "$TRAIN_MODE" in
    overwrite)
        MODE_ARG="--overwrite"
        ;;
    resume)
        MODE_ARG="--resume"
        ;;
    *)
        echo "TRAIN_MODE must be 'overwrite' or 'resume', got: $TRAIN_MODE" >&2
        exit 1
        ;;
esac

echo "=== training $CONFIG_NAME ($TRAIN_MODE) ==="
python scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    "$MODE_ARG"

kill "$UPLOADER_PID" 2>/dev/null || true
wait "$UPLOADER_PID" 2>/dev/null || true
UPLOADER_PID=""
trap - EXIT

echo "=== uploading remaining checkpoints ==="
python scripts/oss_ckpt_uploader.py \
    --local-run-dir "$RUN_DIR" \
    --oss-run-dir "$OSS_RUN_DIR" \
    --poll-secs 30 \
    --settle-secs 120 \
    --max-idle-polls 3

echo "=== training and checkpoint delivery complete ==="
