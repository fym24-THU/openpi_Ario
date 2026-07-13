"""Evaluate checkpoint on training set.

Loads the pi05_xingchen_fold_ario_debug checkpoint and evaluates prediction
accuracy on the training episodes. The metric is the L1/L2 distance between
the predicted next-step position and the actual next-frame state.
"""
"""
# 确保设置好 OSS 凭证环境变量
export AWS_ACCESS_KEY_ID=$ALIBABA_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$ALIBABA_ACCESS_KEY_SECRET

# 评估最后一个 checkpoint (step 500)
python scripts/eval_xingchen_train.py

# 评估其他 checkpoint
python scripts/eval_xingchen_train.py --checkpoint_dir ./checkpoints/pi05_xingchen_fold_ario_debug/pi05_xingchen_fold_ario_debug/100

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
        default="./checkpoints/pi05_xingchen_fold_ario_debug/pi05_xingchen_fold_ario_debug/500",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_xingchen_fold_ario_debug",
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
    )
    action_horizon = train_config.model.action_horizon

    print("Building dataset...")
    dataset = ArioStreamingDataset(ario_cfg, action_horizon=action_horizon)
    total_frames = len(dataset)
    print(f"Total frames in dataset: {total_frames}")

    num_samples = min(args.num_samples, total_frames - 1)
    indices = np.random.choice(total_frames - 1, size=num_samples, replace=False)
    indices.sort()

    # Evaluate: for each sample, predict actions and compare first predicted action
    # (which represents the predicted next state) against actual next-frame state.
    l1_errors = []
    l2_errors = []
    per_dim_l1 = []

    print(f"\nEvaluating {num_samples} samples...")
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[int(idx)]
        next_sample = dataset[int(idx) + 1]

        obs = {
            "observation/image": sample["observation/image"],
            "observation/state": sample["observation/state"],
            "prompt": sample["prompt"],
        }

        result = policy.infer(obs)
        # result["actions"] shape: (action_horizon, 31) - absolute actions after output transforms
        predicted_actions = result["actions"]  # (action_horizon, 31)
        predicted_next_state = predicted_actions[0]  # first step prediction

        # Ground truth: next frame's state
        gt_next_state = next_sample["observation/state"]

        # Compute errors
        diff = predicted_next_state - gt_next_state
        l1 = np.abs(diff).mean()
        l2 = np.sqrt((diff**2).mean())

        l1_errors.append(l1)
        l2_errors.append(l2)
        per_dim_l1.append(np.abs(diff))

    # Summary statistics
    l1_errors = np.array(l1_errors)
    l2_errors = np.array(l2_errors)
    per_dim_l1 = np.stack(per_dim_l1, axis=0)  # (num_samples, 31)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Num samples: {num_samples}")
    print(f"Total frames: {total_frames}")
    print("-" * 60)
    print(f"Mean L1 error (avg over dims): {l1_errors.mean():.6f}")
    print(f"Std  L1 error:                 {l1_errors.std():.6f}")
    print(f"Mean L2 error (RMS over dims): {l2_errors.mean():.6f}")
    print(f"Std  L2 error:                 {l2_errors.std():.6f}")
    print(f"Median L1 error:               {np.median(l1_errors):.6f}")
    print(f"Max L1 error:                  {l1_errors.max():.6f}")
    print("-" * 60)

    # Per-dimension breakdown
    dim_labels = [
        # eef_torso: 9 dims
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5", "torso_6", "torso_7", "torso_8",
        # head: 2 dims
        "head_0", "head_1",
        # eef_left: 9 dims
        "left_0", "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8",
        # gripper_cmd left: 1 dim
        "grip_L",
        # eef_right: 9 dims
        "right_0", "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8",
        # gripper_cmd right: 1 dim
        "grip_R",
    ]

    print("\nPer-dimension mean L1 error:")
    mean_per_dim = per_dim_l1.mean(axis=0)
    for d in range(min(len(dim_labels), len(mean_per_dim))):
        bar = "#" * int(mean_per_dim[d] * 200)
        print(f"  {dim_labels[d]:>8s}: {mean_per_dim[d]:.6f}  {bar}")

    # Save results
    results_path = Path(args.checkpoint_dir) / "eval_results.npz"
    np.savez(
        results_path,
        l1_errors=l1_errors,
        l2_errors=l2_errors,
        per_dim_l1=per_dim_l1,
        indices=indices,
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
