#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_PARENT="${OPENPI_SNAPSHOT_PARENT:-$HOME/.cache/openpi/sslaunch-snapshots}"
SNAPSHOT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/$SNAPSHOT_ID"

if (($# == 0)); then
    cat >&2 <<'EOF'
Usage:
  bash scripts/submit_xingchen_train.sh [sslaunch submit options]

Example:
  bash scripts/submit_xingchen_train.sh \
    -c zhongwei -q embody -j xingchen-train-openpi -n 1 \
    -e ALIBABA_ACCESS_KEY_ID="$ALIBABA_ACCESS_KEY_ID" \
    -e ALIBABA_ACCESS_KEY_SECRET="$ALIBABA_ACCESS_KEY_SECRET"
EOF
    exit 2
fi

mkdir -p "$SNAPSHOT_ROOT"

cleanup_failed_snapshot() {
    rm -rf -- "$SNAPSHOT_ROOT"
}
trap cleanup_failed_snapshot ERR INT TERM

echo "Creating immutable source snapshot: $SNAPSHOT_ROOT"
tar \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./assets' \
    --exclude='./checkpoints' \
    --exclude='./data' \
    --exclude='./wandb' \
    --exclude='./logs' \
    --exclude='./job-logs' \
    --exclude='./__pycache__' \
    --exclude='*/__pycache__' \
    --exclude='./.ruff_cache' \
    --exclude='./.pytest_cache' \
    -C "$PROJECT_ROOT" -cf - . \
    | tar -C "$SNAPSHOT_ROOT" -xf -

# Runtime state remains shared and is intentionally not duplicated.
for shared_dir in .venv assets checkpoints data; do
    if [[ -e "$PROJECT_ROOT/$shared_dir" ]]; then
        ln -s "$PROJECT_ROOT/$shared_dir" "$SNAPSHOT_ROOT/$shared_dir"
    fi
done

mkdir -p "$SNAPSHOT_ROOT/.openpi-snapshot"
{
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_root=$PROJECT_ROOT"
    echo "snapshot_root=$SNAPSHOT_ROOT"
    if git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    else
        echo "git_commit=unavailable"
    fi
} >"$SNAPSHOT_ROOT/.openpi-snapshot/metadata.txt"

if git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$PROJECT_ROOT" status --short >"$SNAPSHOT_ROOT/.openpi-snapshot/git-status.txt"
    git -C "$PROJECT_ROOT" diff --binary HEAD >"$SNAPSHOT_ROOT/.openpi-snapshot/working-tree.patch"
fi

echo "Submitting frozen snapshot..."
sslaunch submit "$@" -- \
    env \
    "OPENPI_ROOT=$SNAPSHOT_ROOT" \
    "OPENPI_SNAPSHOT_ROOT=$SNAPSHOT_ROOT" \
    bash "$SNAPSHOT_ROOT/scripts/run_xingchen_train.sh"

# From this point the submitted job owns the snapshot and removes it on exit.
trap - ERR INT TERM
echo "Snapshot retained until the job exits: $SNAPSHOT_ROOT"
