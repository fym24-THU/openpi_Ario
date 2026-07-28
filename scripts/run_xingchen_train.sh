#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ "${OPENPI_CONTAINER:-0}" != "1" ]]; then
    if [[ ! -f .venv/bin/activate ]]; then
        echo "Missing .venv. Run 'uv sync', or submit with the openpi training container." >&2
        exit 1
    fi
    source .venv/bin/activate
fi

# Prefer the source tree mounted by sslaunch over the snapshot baked into the
# image, so submitted jobs always run the current server-side code.
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

# Accept either Alibaba-named credentials or AWS-compatible credentials
# injected by the cluster. Never store credentials in this script or image.
if [[ -n "${ALIBABA_ACCESS_KEY_ID:-}" ]]; then
    export AWS_ACCESS_KEY_ID="$ALIBABA_ACCESS_KEY_ID"
fi
if [[ -n "${ALIBABA_ACCESS_KEY_SECRET:-}" ]]; then
    export AWS_SECRET_ACCESS_KEY="$ALIBABA_ACCESS_KEY_SECRET"
fi
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID or ALIBABA_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY or ALIBABA_ACCESS_KEY_SECRET}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Prefer the CUDA 12.9 ptxas installed in this Python environment over an
# older system CUDA toolkit. Do not set xla_gpu_cuda_data_dir to this path:
# the nvidia-cuda-nvcc wheel provides ptxas, not a complete CUDA toolkit.
CUDA_NVCC_BIN="$(python - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.cuda_nvcc")
if spec is None or spec.submodule_search_locations is None:
    raise RuntimeError(
        "nvidia-cuda-nvcc-cu12 is missing; install version 12.9.86 in .venv"
    )
print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "bin")
PY
)"
export PATH="$CUDA_NVCC_BIN:$PATH"

echo "=== Runtime check ==="
python - <<'PY'
import importlib.metadata
import shutil
import subprocess

import jax

expected_versions = {
    "jax": "0.5.3",
    "jaxlib": "0.5.3",
    "jax-cuda12-plugin": "0.5.3",
    "jax-cuda12-pjrt": "0.5.3",
    "nvidia-cuda-nvcc-cu12": "12.9.86",
}
for package, expected in expected_versions.items():
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"Expected {package}=={expected}, got {actual}")

ptxas = shutil.which("ptxas")
if ptxas is None:
    raise RuntimeError("ptxas was not found on PATH")
ptxas_version = subprocess.run(
    [ptxas, "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout
if "release 12.9" not in ptxas_version:
    raise RuntimeError(f"Expected CUDA 12.9 ptxas, got:\n{ptxas_version}")

print("JAX:", jax.__version__)
print("ptxas:", ptxas)
print(ptxas_version.strip())
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise RuntimeError("JAX did not discover a GPU backend")
PY

echo "=== Step 1: Compute norm stats (if not already done) ==="
NORM_STATS_PATH="./assets/pi05_xingchen_ario/xingchen/new_blocks/norm_stats.json"
if [ -f "$NORM_STATS_PATH" ]; then
    echo "Norm stats already exist at $NORM_STATS_PATH, skipping."
else
    echo "Computing normalization statistics for new data..."
    python scripts/compute_norm_stats.py --config-name pi05_xingchen_ario
fi

echo "=== Step 2: Training (JAX) ==="
python scripts/train.py pi05_xingchen_ario --overwrite

echo "=== Done ==="
