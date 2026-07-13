#!/usr/bin/env python3
"""Extract loss from training log and plot the loss curve."""
"""python scripts/plot_loss.py <job-id>"""

import re
import subprocess
import sys

import matplotlib.pyplot as plt


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "fanyiming-xingchen-n1-260710094438"

    print(f"Fetching log for job: {job_id}")
    result = subprocess.run(
        ["sslaunch", "job", "log", job_id],
        capture_output=True, text=True
    )
    log_text = result.stdout + result.stderr

    pattern = re.compile(r"step=(\d+)\s+loss=([\d.]+)")
    steps, losses = [], []
    for match in pattern.finditer(log_text):
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))

    if not steps:
        print("No loss data found in log.")
        sys.exit(1)

    print(f"Found {len(steps)} data points (step {steps[0]} to {steps[-1]})")

    plt.figure(figsize=(12, 5))
    plt.plot(steps, losses, linewidth=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss — {job_id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"logs/loss_curve_{job_id}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
