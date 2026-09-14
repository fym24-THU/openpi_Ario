"""Offline open-loop evaluation for Songling 14-D qpos checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import dataclasses
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from openpi.datasets.ario_dataset import ArioStreamingDataset
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config

plt.switch_backend("Agg")

ACTION_DIM = 14
DIMENSION_NAMES = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)
DIMENSION_UNITS = ("rad",) * 6 + ("raw_qpos",) + ("rad",) * 6 + ("raw_qpos",)
DIMENSION_GROUPS = (
    ("Left arm joints", tuple(range(6))),
    ("Left gripper", (6,)),
    ("Right arm joints", tuple(range(7, 13))),
    ("Right gripper", (13,)),
)


@dataclasses.dataclass
class RmseAccumulator:
    horizon: int
    squared_error_sum: np.ndarray = dataclasses.field(init=False)
    max_squared_error: np.ndarray = dataclasses.field(init=False)
    counts: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.squared_error_sum = np.zeros((self.horizon, ACTION_DIM), dtype=np.float64)
        self.max_squared_error = np.zeros((self.horizon, ACTION_DIM), dtype=np.float64)
        self.counts = np.zeros(self.horizon, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}")
        if prediction.ndim != 2 or prediction.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected [H, {ACTION_DIM}] arrays, got {prediction.shape}")
        if not 0 < len(prediction) <= self.horizon:
            raise ValueError(f"Valid horizon must be in [1, {self.horizon}], got {len(prediction)}")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Prediction or target contains NaN/Inf")

        squared_error = np.square(prediction - target)
        valid_horizon = len(prediction)
        self.squared_error_sum[:valid_horizon] += squared_error
        self.max_squared_error[:valid_horizon] = np.maximum(
            self.max_squared_error[:valid_horizon],
            squared_error,
        )
        self.counts[:valid_horizon] += 1

    def results(self) -> tuple[np.ndarray, np.ndarray]:
        denominator = self.counts[:, None]
        mean_rmse = np.sqrt(
            np.divide(
                self.squared_error_sum,
                denominator,
                out=np.full_like(self.squared_error_sum, np.nan),
                where=denominator > 0,
            )
        )
        max_rmse = np.sqrt(np.where(denominator > 0, self.max_squared_error, np.nan))
        return mean_rmse, max_rmse


def resolve_checkpoint(path: Path) -> Path:
    """Accept either a concrete checkpoint step or a run directory."""
    path = path.resolve()
    if (path / "params").is_dir():
        return path
    steps = sorted(
        (candidate for candidate in path.iterdir() if candidate.is_dir() and candidate.name.isdigit()),
        key=lambda candidate: int(candidate.name),
    )
    if not steps:
        raise FileNotFoundError(f"No checkpoint step containing params found under {path}")
    checkpoint = steps[-1]
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Latest checkpoint has no params directory: {checkpoint}")
    return checkpoint


def candidate_indices(episode_lengths: Sequence[int], stride: int) -> np.ndarray:
    """Return global frame indices without terminal frames or cross-episode pairs."""
    if stride <= 0:
        raise ValueError("--stride must be positive")
    indices = []
    offset = 0
    for length in episode_lengths:
        indices.extend(range(offset, offset + max(0, length - 1), stride))
        offset += length
    return np.asarray(indices, dtype=np.int64)


def select_indices(
    episode_lengths: Sequence[int],
    *,
    stride: int,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    indices = candidate_indices(episode_lengths, stride)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        if max_samples < len(indices):
            indices = np.random.default_rng(seed).choice(indices, size=max_samples, replace=False)
    return np.sort(indices)


def write_csv(
    output_path: Path,
    mean_rmse: np.ndarray,
    max_rmse: np.ndarray,
    counts: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("chunk_step", "dimension_index", "dimension", "unit", "mean_rmse", "max_rmse", "samples"))
        for step in range(len(mean_rmse)):
            for dimension in range(ACTION_DIM):
                writer.writerow(
                    (
                        step + 1,
                        dimension,
                        DIMENSION_NAMES[dimension],
                        DIMENSION_UNITS[dimension],
                        mean_rmse[step, dimension],
                        max_rmse[step, dimension],
                        int(counts[step]),
                    )
                )


def plot_rmse(output_path: Path, mean_rmse: np.ndarray, max_rmse: np.ndarray) -> None:
    steps = np.arange(1, len(mean_rmse) + 1)
    figure, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)
    figure.suptitle("Songling open-loop error by action chunk step", fontsize=16)

    for row, (group_name, dimensions) in enumerate(DIMENSION_GROUPS):
        unit = DIMENSION_UNITS[dimensions[0]]
        for column, (values, metric_name) in enumerate(((mean_rmse, "Mean RMSE"), (max_rmse, "Max RMSE"))):
            axis = axes[row, column]
            for dimension in dimensions:
                axis.plot(steps, values[:, dimension], label=DIMENSION_NAMES[dimension], linewidth=1.8)
            axis.set(
                title=f"{group_name} — {metric_name}",
                xlabel="Chunk step",
                ylabel=f"{metric_name} ({unit})",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_observation(sample: dict) -> dict:
    return {
        "observation/image": sample["observation/image"],
        "observation/cam_high": sample["observation/cam_high"],
        "observation/cam_left_wrist": sample["observation/cam_left_wrist"],
        "observation/cam_right_wrist": sample["observation/cam_right_wrist"],
        "observation/state": sample["observation/state"],
        "prompt": sample["prompt"],
    }


def json_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [[float(value) if np.isfinite(value) else None for value in row] for row in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("offline_eval_results/songling"))
    parser.add_argument(
        "--trace-output",
        type=Path,
        help=(
            "Optional compressed NPZ with per-sample observation states, predictions, "
            "targets, and frame indices for slave-side replay."
        ),
    )
    parser.add_argument("--max-samples", type=int, help="Random sample limit; default evaluates every valid frame")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--chunk-horizon", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint(args.checkpoint_dir)
    train_config = get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.ario_config is None:
        raise ValueError(f"Config {args.config_name!r} is not an Ario streaming config")

    ario_config = data_config.ario_config
    if args.max_episodes is not None:
        ario_config = dataclasses.replace(ario_config, max_episodes=args.max_episodes)
    model_horizon = train_config.model.action_horizon
    chunk_horizon = args.chunk_horizon or model_horizon
    if not 0 < chunk_horizon <= model_horizon:
        raise ValueError(f"--chunk-horizon must be in [1, {model_horizon}]")

    print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
    policy = create_trained_policy(train_config, checkpoint_dir, pytorch_device=args.device)
    print("Building training dataset...", flush=True)
    dataset = ArioStreamingDataset(ario_config, action_horizon=model_horizon)
    episode_lengths = dataset.episode_lengths
    indices = select_indices(
        episode_lengths,
        stride=args.stride,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not len(indices):
        raise ValueError("No non-terminal evaluation samples were found")

    accumulator = RmseAccumulator(chunk_horizon)
    rng = np.random.default_rng(args.seed)
    inference_times = []
    trace_predictions = None
    trace_targets = None
    trace_states = None
    trace_valid_horizons = None
    trace_episode_indices = None
    trace_frame_indices = None
    if args.trace_output is not None:
        trace_shape = (len(indices), chunk_horizon, ACTION_DIM)
        trace_predictions = np.full(trace_shape, np.nan, dtype=np.float32)
        trace_targets = np.full(trace_shape, np.nan, dtype=np.float32)
        trace_states = np.empty((len(indices), ACTION_DIM), dtype=np.float32)
        trace_valid_horizons = np.empty(len(indices), dtype=np.int32)
        trace_episode_indices = np.empty(len(indices), dtype=np.int32)
        trace_frame_indices = np.empty(len(indices), dtype=np.int64)
    started = time.monotonic()
    for trace_index, index in enumerate(tqdm(indices, desc="Open-loop evaluation")):
        episode_index, frame_index = dataset.global_to_local(int(index))
        valid_horizon = min(chunk_horizon, episode_lengths[episode_index] - frame_index - 1)
        # Fail instead of silently substituting a random frame, which would corrupt alignment.
        sample = dataset.get_item(int(index), retries=1)
        noise = rng.standard_normal((model_horizon, train_config.model.action_dim), dtype=np.float32)

        inference_started = time.monotonic()
        result = policy.infer(make_observation(sample), noise=noise)
        inference_times.append((time.monotonic() - inference_started) * 1000)

        prediction = np.asarray(result["actions"][:valid_horizon, :ACTION_DIM])
        target = np.asarray(sample["actions"][:valid_horizon, :ACTION_DIM])
        accumulator.update(prediction, target)
        if trace_predictions is not None:
            trace_predictions[trace_index, :valid_horizon] = prediction
            trace_targets[trace_index, :valid_horizon] = target
            trace_states[trace_index] = np.asarray(sample["observation/state"][:ACTION_DIM])
            trace_valid_horizons[trace_index] = valid_horizon
            trace_episode_indices[trace_index] = episode_index
            trace_frame_indices[trace_index] = frame_index

    mean_rmse, max_rmse = accumulator.results()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rmse_by_chunk_step.csv", mean_rmse, max_rmse, accumulator.counts)
    plot_rmse(args.output_dir / "rmse_by_chunk_step.png", mean_rmse, max_rmse)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "config_name": args.config_name,
        "sample_count": len(indices),
        "episode_count": len(episode_lengths),
        "chunk_horizon": chunk_horizon,
        "stride": args.stride,
        "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started,
        "mean_inference_ms": float(np.mean(inference_times)),
        "samples_by_chunk_step": accumulator.counts.tolist(),
        "dimension_names": DIMENSION_NAMES,
        "dimension_units": DIMENSION_UNITS,
        "mean_rmse": json_matrix(mean_rmse),
        "max_rmse": json_matrix(max_rmse),
        "evaluated_global_indices": indices.tolist(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_output,
            predictions=trace_predictions,
            targets=trace_targets,
            observation_states=trace_states,
            valid_horizons=trace_valid_horizons,
            global_indices=indices,
            episode_indices=trace_episode_indices,
            frame_indices=trace_frame_indices,
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "checkpoint_dir": str(checkpoint_dir),
                        "config_name": args.config_name,
                        "chunk_horizon": chunk_horizon,
                        "action_dim": ACTION_DIM,
                        "stride": args.stride,
                        "seed": args.seed,
                    },
                    ensure_ascii=False,
                )
            ),
        )
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"CSV:     {args.output_dir / 'rmse_by_chunk_step.csv'}")
    print(f"Plot:    {args.output_dir / 'rmse_by_chunk_step.png'}")
    if args.trace_output is not None:
        print(f"Trace:   {args.trace_output}")


if __name__ == "__main__":
    main()
