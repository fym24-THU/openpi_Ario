#!/usr/bin/env python3
"""Extract loss from a training log file and plot the loss curve.

Usage:
    python scripts/plot_loss_local.py <log_file> [output.png]
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_loss_local.py <log_file> [output.png]")
        sys.exit(1)

    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"File not found: {log_file}")
        sys.exit(1)

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else log_file.with_suffix(".png")

    print(f"Reading: {log_file}")
    text = log_file.read_text()

    pattern = re.compile(r"Step (\d+):.*?loss=([\d.]+)")
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
    plt.title(f"Training Loss — {log_file.name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
