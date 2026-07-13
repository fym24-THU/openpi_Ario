#!/usr/bin/env python3
"""Extract loss from local torchrun log files and plot the loss curve."""
"""Usage: 
python scripts/plot_loss_local.py            # 自动找最新日志
python scripts/plot_loss_local.py logs/none_lb7mn9x2  # 指定某次运行的目录
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def find_latest_log_dir(base: Path) -> Path:
    """Find the most recent torchrun log directory."""
    candidates = sorted(base.glob("none_*/attempt_0/0/stderr.log"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"No log files found under {base}")
        sys.exit(1)
    return candidates[-1]


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")

    # If user passed a directory containing stderr.log directly, use it
    if (base / "stderr.log").exists():
        log_file = base / "stderr.log"
    elif (base / "attempt_0/0/stderr.log").exists():
        log_file = base / "attempt_0/0/stderr.log"
    else:
        log_file = find_latest_log_dir(base)

    print(f"Reading: {log_file}")
    text = log_file.read_text()

    pattern = re.compile(r"step=(\d+)\s+loss=([\d.]+)")
    steps, losses = [], []
    for match in pattern.finditer(text):
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))

    if not steps:
        print("No loss data found.")
        sys.exit(1)

    print(f"Found {len(steps)} data points (step {steps[0]} to {steps[-1]})")

    plt.figure(figsize=(12, 5))
    plt.plot(steps, losses, linewidth=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (local log)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = base / "loss_curve.png" if base != Path("logs") else Path("logs/loss_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
