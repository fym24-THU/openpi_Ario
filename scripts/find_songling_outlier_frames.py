"""Find Songling frames that create extreme training actions.

The scanner uses the same episode selection, canonical55 mapping, action offset,
and action horizon as training. It reads state.pt files only; videos are never
downloaded.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from openpi.datasets.ario_dataset import ArioStreamingDataset
from openpi.training.config import get_config

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


@dataclasses.dataclass(frozen=True)
class Finding:
    kind: str
    episode_uri: str
    dimension: int
    source_frame: int
    target_frame: int
    horizon: int
    source_value: float
    target_value: float
    delta: float

    @property
    def score(self) -> float:
        value = self.source_value if self.kind == "absolute_value" else self.delta
        return abs(value) if math.isfinite(value) else math.inf


class TopFindings:
    """Keep the globally largest findings for each category and dimension."""

    def __init__(self, top_k: int):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._top_k = top_k
        self._serial = 0
        self._heaps: dict[tuple[str, int], list[tuple[float, int, Finding]]] = {}

    def add(self, finding: Finding) -> None:
        key = (finding.kind, finding.dimension)
        heap = self._heaps.setdefault(key, [])
        item = (finding.score, self._serial, finding)
        self._serial += 1
        if len(heap) < self._top_k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def sorted_findings(self) -> list[Finding]:
        findings = [item[2] for heap in self._heaps.values() for item in heap]
        return sorted(findings, key=lambda item: (item.kind, item.dimension, -item.score))


def _largest_flat_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    """Return flat indices of the largest absolute values, treating NaN/Inf as most anomalous."""
    scores = np.abs(np.asarray(values)).reshape(-1)
    scores = np.where(np.isfinite(scores), scores, np.inf)
    count = min(top_k, scores.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == scores.size:
        return np.arange(scores.size)
    return np.argpartition(scores, -count)[-count:]


def scan_episode(
    qpos: np.ndarray,
    *,
    episode_uri: str,
    dimensions: tuple[int, ...],
    action_horizon: int,
    action_start_offset: int,
    top_k: int,
) -> list[Finding]:
    """Return per-episode candidates for raw values, adjacent jumps, and training deltas."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos shape [frames, dimensions], got {qpos.shape}")
    if not dimensions:
        raise ValueError("At least one dimension is required")
    if min(dimensions) < 0 or max(dimensions) >= qpos.shape[1]:
        raise ValueError(f"Dimensions {dimensions} are invalid for qpos shape {qpos.shape}")

    findings: list[Finding] = []
    frame_count = len(qpos)

    for dim in dimensions:
        for frame in _largest_flat_indices(qpos[:, dim], top_k):
            value = float(qpos[frame, dim])
            findings.append(
                Finding(
                    kind="absolute_value",
                    episode_uri=episode_uri,
                    dimension=dim,
                    source_frame=int(frame),
                    target_frame=int(frame),
                    horizon=0,
                    source_value=value,
                    target_value=value,
                    delta=0.0,
                )
            )

    if frame_count >= 2:
        adjacent_deltas = np.diff(qpos[:, dimensions], axis=0)
        for local_dim, dim in enumerate(dimensions):
            for source_frame in _largest_flat_indices(adjacent_deltas[:, local_dim], top_k):
                target_frame = int(source_frame + 1)
                findings.append(
                    Finding(
                        kind="adjacent_jump",
                        episode_uri=episode_uri,
                        dimension=dim,
                        source_frame=int(source_frame),
                        target_frame=target_frame,
                        horizon=1,
                        source_value=float(qpos[source_frame, dim]),
                        target_value=float(qpos[target_frame, dim]),
                        delta=float(adjacent_deltas[source_frame, local_dim]),
                    )
                )

    if frame_count and action_horizon > 0:
        offsets = action_start_offset + np.arange(action_horizon)
        source_frames = np.arange(frame_count)
        target_frames = np.minimum(source_frames[:, None] + offsets[None, :], frame_count - 1)
        selected_qpos = qpos[:, dimensions]
        action_deltas = qpos[target_frames][:, :, dimensions] - selected_qpos[:, None, :]

        for local_dim, dim in enumerate(dimensions):
            values = action_deltas[:, :, local_dim]
            for flat_index in _largest_flat_indices(values, top_k):
                source_frame, action_step = np.unravel_index(flat_index, values.shape)
                target_frame = int(target_frames[source_frame, action_step])
                findings.append(
                    Finding(
                        kind="training_delta",
                        episode_uri=episode_uri,
                        dimension=dim,
                        source_frame=int(source_frame),
                        target_frame=target_frame,
                        horizon=int(offsets[action_step]),
                        source_value=float(qpos[source_frame, dim]),
                        target_value=float(qpos[target_frame, dim]),
                        delta=float(values[source_frame, action_step]),
                    )
                )

    return findings


def _default_dimensions(data_config: Any) -> tuple[int, ...]:
    stats = data_config.norm_stats.get("actions") if data_config.norm_stats else None
    if stats is None or stats.q01 is None or stats.q99 is None:
        return tuple(range(len(DIMENSION_NAMES)))
    widths = np.asarray(stats.q99) - np.asarray(stats.q01)
    degenerate = tuple(int(index) for index in np.flatnonzero(widths <= 1e-6))
    return degenerate or tuple(range(min(len(widths), len(DIMENSION_NAMES))))


def _write_csv(path: Path, findings: list[Finding], frame_stride: int) -> None:
    fieldnames = [
        "kind",
        "score",
        "episode_uri",
        "dimension",
        "dimension_name",
        "source_frame",
        "target_frame",
        "source_raw_frame",
        "target_raw_frame",
        "horizon",
        "source_value",
        "target_value",
        "delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    **dataclasses.asdict(finding),
                    "score": finding.score,
                    "dimension_name": DIMENSION_NAMES[finding.dimension],
                    "source_raw_frame": finding.source_frame * frame_stride,
                    "target_raw_frame": finding.target_frame * frame_stride,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_songling_garment_folding_ario")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        help="14-D action dimensions to scan. Defaults to dimensions with collapsed quantile ranges.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Findings retained per category and dimension.")
    parser.add_argument("--max-episodes", type=int, help="Only scan the first N selected episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("outlier_reports/songling"))
    args = parser.parse_args()

    train_config = get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.ario_config is None:
        raise ValueError(f"Config {args.config_name!r} is not an Ario dataset")
    if data_config.ario_config.data_format != "songling_canonical55":
        raise ValueError(
            f"Config {args.config_name!r} uses {data_config.ario_config.data_format!r}, not songling_canonical55"
        )

    dimensions = tuple(args.dimensions) if args.dimensions else _default_dimensions(data_config)
    ario_config = dataclasses.replace(
        data_config.ario_config,
        skip_video=True,
        max_episodes=args.max_episodes,
    )
    print(f"Scanning dimensions: {dimensions}")
    if data_config.norm_stats and (stats := data_config.norm_stats.get("actions")) is not None:
        widths = np.asarray(stats.q99) - np.asarray(stats.q01)
        print("Configured action q99-q01:", {dim: float(widths[dim]) for dim in dimensions})

    dataset = ArioStreamingDataset(ario_config, train_config.model.action_horizon)
    top = TopFindings(args.top_k)
    frame_count = 0
    failed_episodes: list[dict[str, str]] = []

    episodes = dataset._episodes  # noqa: SLF001 -- diagnostics need episode identities.
    for bucket, prefix in tqdm(episodes, desc="Scanning state trajectories"):
        episode_uri = f"s3://{bucket}/{prefix}"
        try:
            qpos, _ = dataset._get_episode(bucket, prefix, f"{bucket}/{prefix}")  # noqa: SLF001
            frame_count += len(qpos)
            for finding in scan_episode(
                qpos,
                episode_uri=episode_uri,
                dimensions=dimensions,
                action_horizon=train_config.model.action_horizon,
                action_start_offset=ario_config.action_start_offset,
                top_k=args.top_k,
            ):
                top.add(finding)
        except Exception as error:
            failed_episodes.append({"episode_uri": episode_uri, "error": repr(error)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    findings = top.sorted_findings()
    csv_path = args.output_dir / "outliers.csv"
    summary_path = args.output_dir / "summary.json"
    _write_csv(csv_path, findings, ario_config.video_downsample_rate)
    summary_path.write_text(
        json.dumps(
            {
                "config_name": args.config_name,
                "dimensions": dimensions,
                "episodes_scanned": len(episodes) - len(failed_episodes),
                "frames_scanned": frame_count,
                "action_horizon": train_config.model.action_horizon,
                "action_start_offset": ario_config.action_start_offset,
                "top_k_per_category_and_dimension": args.top_k,
                "failed_episodes": failed_episodes,
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(findings)} findings to {csv_path}")
    print(f"Wrote scan summary to {summary_path}")


if __name__ == "__main__":
    main()
