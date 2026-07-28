"""Evaluate checkpoint on training set.

Loads the pi05_xingchen_bench_xc03_pp3_pp10 checkpoint and evaluates prediction
accuracy on the training episodes. The metric is the L1/L2 distance between
the predicted next-step position and the actual next-frame state.
"""
"""
# 确保设置好 OSS 凭证环境变量
export AWS_ACCESS_KEY_ID=$ALIBABA_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$ALIBABA_ACCESS_KEY_SECRET

# 评估 checkpoint
python scripts/eval_xingchen_train.py

# 增加样本数
python scripts/eval_xingchen_train.py --num_samples 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on training set")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/pi05_xingchen_bench_xc03_pp3_pp10/bench_xc03_pp3_pp10",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_xingchen_bench_xc03_pp3_pp10",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    from openpi.datasets.ario_dataset import ArioConfig, ArioStreamingDataset
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    # Load config and policy
    print(f"Loading config: {args.config_name}")
    train_config = get_config(args.config_name)

    print(f"Loading checkpoint: {args.checkpoint_dir}")
    policy = create_trained_policy(train_config, args.checkpoint_dir, pytorch_device=args.device)

    # Build dataset with same config as training
    data_cfg = train_config.data
    ario_cfg = ArioConfig(
        s3_prefixes=data_cfg.s3_prefixes,
        s3_endpoint=data_cfg.s3_endpoint,
        video_downsample_rate=data_cfg.video_downsample_rate,
        min_frames=data_cfg.min_frames,
        image_size=data_cfg.image_size,
        task=data_cfg.default_prompt,
        cache_size=32,
        max_episodes=data_cfg.max_episodes,
        disk_cache_dir=data_cfg.disk_cache_dir,
        disk_cache_max_gb=data_cfg.disk_cache_max_gb,
        instruction_key=data_cfg.instruction_key,
    )
    action_horizon = train_config.model.action_horizon

    print("Building dataset...")
    dataset = ArioStreamingDataset(ario_cfg, action_horizon=action_horizon)
    total_frames = len(dataset)
    print(f"Total frames in dataset: {total_frames}")

    num_samples = min(args.num_samples, total_frames)
    indices = np.random.choice(total_frames, size=num_samples, replace=False)
    indices.sort()

    # Evaluate: predict action chunk and compare against ground truth chunk.
    # dataset["actions"] shape: (action_horizon, 31) — absolute states at future timesteps.
    # policy.infer() output also in absolute space after AbsoluteActions transform.
    chunk_l1_errors = []
    chunk_l2_errors = []
    first_step_l1 = []
    first_step_l2 = []
    per_dim_l1_all = []

    print(f"\nEvaluating {num_samples} samples (action_horizon={action_horizon})...")
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[int(idx)]

        obs = {
            "observation/image": sample["observation/image"],
            "observation/state": sample["observation/state"],
            "prompt": sample["prompt"],
        }

        result = policy.infer(obs)
        predicted = np.array(result["actions"])  # (action_horizon, 31)
        gt = np.array(sample["actions"][:, :predicted.shape[-1]])  # (action_horizon, 31)

        # Full chunk error
        diff = predicted - gt
        chunk_l1 = np.abs(diff).mean()
        chunk_l2 = np.sqrt((diff**2).mean())
        chunk_l1_errors.append(chunk_l1)
        chunk_l2_errors.append(chunk_l2)

        # First step error (actions[0] = current state prediction)
        diff0 = predicted[0] - gt[0]
        first_step_l1.append(np.abs(diff0).mean())
        first_step_l2.append(np.sqrt((diff0**2).mean()))

        # Per-dim L1 averaged over horizon
        per_dim_l1_all.append(np.abs(diff).mean(axis=0))

    chunk_l1_errors = np.array(chunk_l1_errors)
    chunk_l2_errors = np.array(chunk_l2_errors)
    first_step_l1 = np.array(first_step_l1)
    first_step_l2 = np.array(first_step_l2)
    per_dim_l1_all = np.stack(per_dim_l1_all, axis=0)  # (num_samples, 31)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Config:     {args.config_name}")
    print(f"Num samples: {num_samples}")
    print(f"Total frames: {total_frames}")
    print(f"Action horizon: {action_horizon}")
    print("-" * 60)
    print("First-step error (actions[0] vs gt[0]):")
    print(f"  Mean L1: {first_step_l1.mean():.6f}  Std: {first_step_l1.std():.6f}")
    print(f"  Mean L2: {first_step_l2.mean():.6f}  Std: {first_step_l2.std():.6f}")
    print("-" * 60)
    print("Full chunk error (avg over all horizon steps):")
    print(f"  Mean L1: {chunk_l1_errors.mean():.6f}  Std: {chunk_l1_errors.std():.6f}")
    print(f"  Mean L2: {chunk_l2_errors.mean():.6f}  Std: {chunk_l2_errors.std():.6f}")
    print(f"  Median L1: {np.median(chunk_l1_errors):.6f}")
    print(f"  Max L1:    {chunk_l1_errors.max():.6f}")
    print("-" * 60)

    dim_labels = [
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5", "torso_6", "torso_7", "torso_8",
        "head_0", "head_1",
        "left_0", "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8",
        "grip_L",
        "right_0", "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8",
        "grip_R",
    ]

    print("\nPer-dimension mean L1 error (averaged over horizon):")
    mean_per_dim = per_dim_l1_all.mean(axis=0)
    for d in range(min(len(dim_labels), len(mean_per_dim))):
        bar = "#" * int(mean_per_dim[d] * 200)
        print(f"  {dim_labels[d]:>8s}: {mean_per_dim[d]:.6f}  {bar}")

    # Sanity check: if first-step error is very low, model has learned training data
    if first_step_l1.mean() < 0.01:
        print("\n[OK] First-step L1 < 0.01 — model fits training data well.")
    elif first_step_l1.mean() < 0.05:
        print("\n[WARN] First-step L1 in [0.01, 0.05] — partial fit, might need more training.")
    else:
        print("\n[FAIL] First-step L1 > 0.05 — model has NOT learned training data properly.")

    results_path = Path(args.checkpoint_dir) / "eval_results.npz"
    np.savez(
        results_path,
        chunk_l1_errors=chunk_l1_errors,
        chunk_l2_errors=chunk_l2_errors,
        first_step_l1=first_step_l1,
        first_step_l2=first_step_l2,
        per_dim_l1=per_dim_l1_all,
        indices=indices,
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
