#!/bin/bash
set -euo pipefail

if [[ -n "${OPENPI_ROOT:-}" ]]; then
    PROJECT_ROOT="$OPENPI_ROOT"
elif [[ -d "/home/fanyiming/openpi_Ario" ]]; then
    # sslaunch executes a copied script under /mnt/vepfs, while the repository
    # and its virtual environment remain available through the shared home.
    PROJECT_ROOT="/home/fanyiming/openpi_Ario"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    # The venv interpreter is an absolute /home/... symlink. sslaunch mounts
    # that same shared home under /mnt/vepfs/base2/..., so use the equivalent
    # managed Python directly while retaining this venv's site-packages.
    PYTHON_VERSION="$(awk -F= '/^version_info =/ {gsub(/ /, "", $2); print $2}' "$PROJECT_ROOT/.venv/pyvenv.cfg")"
    SHARED_HOME="$(dirname "$PROJECT_ROOT")"
    PROJECT_PYTHON="$PROJECT_ROOT/.venv/python-runtime/bin/python${PYTHON_VERSION}"
    SHARED_PYTHON="$SHARED_HOME/.local/share/uv/python/cpython-${PYTHON_VERSION}-linux-x86_64-gnu/bin/python${PYTHON_VERSION}"
    if [[ -x "$PROJECT_PYTHON" ]]; then
        PYTHON="$PROJECT_PYTHON"
    else
        PYTHON="$SHARED_PYTHON"
    fi
    if [[ ! -x "$PYTHON" ]]; then
        echo "Missing sslaunch Python runtime: $PYTHON" >&2
        echo "Stage Python under .venv/python-runtime before submitting." >&2
        exit 1
    fi
    export PYTHONPATH="$PROJECT_ROOT/.venv/lib/python${PYTHON_VERSION}/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== checking JAX accelerator backend ==="
"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp

backend = jax.default_backend()
devices = jax.devices()
print(f"JAX backend: {backend}")
print(f"JAX devices: {devices}")
if backend != "gpu":
    raise SystemExit(
        "ERROR: JAX GPU backend is unavailable; refusing to train on CPU. "
        "Install the CUDA-enabled JAX dependencies before submitting."
    )

# Seeing the GPU is insufficient on new architectures such as B300. Force a
# small BF16 GEMM through XLA/ptxas so an unsupported toolchain fails now.
@jax.jit
def accelerator_smoke_test(x):
    return x @ x

result = accelerator_smoke_test(jnp.ones((512, 512), dtype=jnp.bfloat16))
result.block_until_ready()
print("JAX BF16 compilation smoke test: PASSED")
PY

: "${CONFIG_NAME:?Set CONFIG_NAME to the training config name}"
EXP_NAME="${EXP_NAME:-$CONFIG_NAME}"
# Match TrainConfig.checkpoint_dir, which resolves sslaunch mount aliases.
RUN_DIR="$(realpath -m "$PROJECT_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME")"

: "${OSS_RUN_ROOT:?Set OSS_RUN_ROOT to an s3+ali:// checkpoint destination}"
OSS_RUN_DIR="${OSS_RUN_ROOT%/}/$CONFIG_NAME/$EXP_NAME"

# Ario reads AWS-compatible names; the uploader reads ALI__-prefixed names.
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${ALIBABA_ACCESS_KEY_ID:-${WRITE__AWS_ACCESS_KEY_ID:-${ALI__AWS_ACCESS_KEY_ID:-}}}}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${ALIBABA_ACCESS_KEY_SECRET:-${WRITE__AWS_SECRET_ACCESS_KEY:-${ALI__AWS_SECRET_ACCESS_KEY:-}}}}"
export ALI__AWS_ACCESS_KEY_ID="${ALI__AWS_ACCESS_KEY_ID:-${WRITE__AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-${ALIBABA_ACCESS_KEY_ID:-}}}}"
export ALI__AWS_SECRET_ACCESS_KEY="${ALI__AWS_SECRET_ACCESS_KEY:-${WRITE__AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-${ALIBABA_ACCESS_KEY_SECRET:-}}}}"
export ALI__OSS_ENDPOINT="${ALI__OSS_ENDPOINT:-https://oss-cn-wulanchabu-internal.aliyuncs.com}"
export ALI__AWS_S3_ADDRESSING_STYLE="${ALI__AWS_S3_ADDRESSING_STYLE:-virtual}"

: "${AWS_ACCESS_KEY_ID:?Set OSS read access key credentials}"
: "${AWS_SECRET_ACCESS_KEY:?Set OSS read secret key credentials}"
: "${ALI__AWS_ACCESS_KEY_ID:?Set OSS upload access key credentials}"
: "${ALI__AWS_SECRET_ACCESS_KEY:?Set OSS upload secret key credentials}"

echo "=== local checkpoints: $RUN_DIR ==="
echo "=== OSS checkpoints:   $OSS_RUN_DIR ==="

UPLOADER_PID=""
cleanup() {
    if [[ -n "$UPLOADER_PID" ]]; then
        kill "$UPLOADER_PID" 2>/dev/null || true
        wait "$UPLOADER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"$PYTHON" scripts/oss_ckpt_uploader.py \
    --local-run-dir "$RUN_DIR" \
    --oss-run-dir "$OSS_RUN_DIR" \
    --poll-secs 60 \
    --settle-secs 10 \
    --max-idle-polls 0 &
UPLOADER_PID=$!

"$PYTHON" scripts/train.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite \
    --no-wandb-enabled

kill "$UPLOADER_PID" 2>/dev/null || true
wait "$UPLOADER_PID" 2>/dev/null || true
UPLOADER_PID=""
trap - EXIT

echo "=== uploading remaining checkpoints ==="
"$PYTHON" scripts/oss_ckpt_uploader.py \
    --local-run-dir "$RUN_DIR" \
    --oss-run-dir "$OSS_RUN_DIR" \
    --poll-secs 30 \
    --settle-secs 10 \
    --max-idle-polls 3

echo "=== training and checkpoint upload complete ==="
