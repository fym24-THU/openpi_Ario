"""Ario streaming dataset that reads directly from OSS without pre-conversion."""

from __future__ import annotations

import io
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ACTION_DIM = 31
PT_FILES = ["eef_torso.pt", "head.pt", "eef_left.pt", "gripper_cmd.pt", "eef_right.pt"]
IMAGE_SIZE = (320, 240)


@dataclass
class ArioConfig:
    s3_prefixes: str = ""
    s3_endpoint: str = "https://oss-cn-wulanchabu-internal.aliyuncs.com"
    video_downsample_rate: int = 6
    min_frames: int = 1885
    image_size: tuple[int, int] = IMAGE_SIZE
    task: str = "fold clothes"
    cache_size: int = 32


class ArioStreamingDataset:
    """Dataset that streams Ario-format episodes directly from S3/OSS.

    Each __getitem__ returns a single training sample with action chunking applied.
    Episodes are cached in an LRU manner to avoid redundant downloads.
    """

    def __init__(self, config: ArioConfig, action_horizon: int):
        self._config = config
        self._action_horizon = action_horizon
        self._s3 = None
        self._pid: int | None = None

        # LRU cache for decoded episodes: ep_key -> (state_action, frames)
        self._cache: OrderedDict[str, tuple[np.ndarray, list[np.ndarray]]] = OrderedDict()
        self._cache_size = config.cache_size

        # Discover episodes and build global frame index
        episodes = self._discover_episodes()
        self._episodes = episodes  # list of (bucket, prefix)

        # Build frame index: for each episode, count usable (downsampled) frames.
        # We need to download .pt to know the length, so do a lightweight pass.
        self._episode_lengths: list[int] = []  # downsampled frame count per episode
        self._cumulative: list[int] = []  # cumulative sum for global index lookup
        self._build_index()

    def _get_s3(self):
        # Recreate client after fork (pid changes)
        pid = os.getpid()
        if self._s3 is None or self._pid != pid:
            import boto3
            from botocore.config import Config

            self._pid = pid
            ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
            sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            cfg = Config(s3={"addressing_style": "virtual"}, signature_version="s3v4")
            kwargs = {}
            if self._config.s3_endpoint:
                kwargs["endpoint_url"] = self._config.s3_endpoint
            self._s3 = boto3.client(
                "s3",
                aws_access_key_id=ak,
                aws_secret_access_key=sk,
                region_name="cn-wulanchabu",
                config=cfg,
                **kwargs,
            )
            self._cache.clear()
        return self._s3

    def _discover_episodes(self) -> list[tuple[str, str]]:
        s3 = self._get_s3()
        episodes = []
        for uri in self._config.s3_prefixes.split(","):
            uri = uri.strip()
            if not uri:
                continue
            bucket, prefix = self._parse_s3_uri(uri)
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/video.mp4"):
                        ep_prefix = key[: key.rfind("/video.mp4") + 1]
                        episodes.append((bucket, ep_prefix))
        episodes.sort(key=lambda x: x[1])
        print(f"ArioStreamingDataset: found {len(episodes)} episodes on S3")
        return episodes

    def _build_index(self):
        """Download eef_torso.pt from each episode to determine its length."""
        from tqdm import tqdm

        s3 = self._get_s3()
        cumulative = 0
        valid_episodes = []
        rate = self._config.video_downsample_rate

        for bucket, prefix in tqdm(self._episodes, desc="Building frame index"):
            try:
                data = self._s3_download_bytes(s3, bucket, prefix + "eef_torso.pt")
                tensor = torch.load(io.BytesIO(data), map_location="cpu")
                raw_len = tensor.shape[0]
            except Exception:
                continue

            if raw_len < self._config.min_frames:
                continue

            n_frames = len(range(0, raw_len, rate))
            valid_episodes.append((bucket, prefix))
            self._episode_lengths.append(n_frames)
            cumulative += n_frames
            self._cumulative.append(cumulative)

        self._episodes = valid_episodes
        print(f"ArioStreamingDataset: {len(self._episodes)} valid episodes, {cumulative} total frames")

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        ep_idx, frame_idx = self._global_to_local(index)
        bucket, prefix = self._episodes[ep_idx]
        cache_key = prefix

        state_action, frames = self._get_episode(bucket, prefix, cache_key)

        # Current frame image and state
        image = frames[frame_idx]
        state = state_action[frame_idx]

        # Action chunk: action_horizon consecutive frames, clamp at end
        actions = self._get_action_chunk(state_action, frame_idx)

        return {
            "observation/image": image,
            "observation/state": state,
            "actions": actions,
            "prompt": self._config.task,
        }

    def _global_to_local(self, index: int) -> tuple[int, int]:
        """Convert global frame index to (episode_idx, local_frame_idx)."""
        import bisect
        ep_idx = bisect.bisect_right(self._cumulative, index)
        local = index - (self._cumulative[ep_idx - 1] if ep_idx > 0 else 0)
        return ep_idx, local

    def _get_action_chunk(self, state_action: np.ndarray, frame_idx: int) -> np.ndarray:
        """Get action_horizon consecutive actions starting at frame_idx, clamping at end."""
        n = len(state_action)
        indices = [min(frame_idx + i, n - 1) for i in range(self._action_horizon)]
        return state_action[indices]

    def _get_episode(
        self, bucket: str, prefix: str, cache_key: str
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get episode data, using LRU cache."""
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Download and decode
        s3 = self._get_s3()
        state_action = self._build_state_action(s3, bucket, prefix)
        frames = self._extract_video_frames(s3, bucket, prefix + "video.mp4")

        # Align lengths and downsample
        raw_len = min(len(state_action), len(frames))
        rate = self._config.video_downsample_rate
        indices = list(range(0, raw_len, rate))

        state_action = state_action[indices].astype(np.float32)
        frames = [frames[i] for i in indices]

        self._cache[cache_key] = (state_action, frames)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return state_action, frames

    def _build_state_action(self, s3, bucket: str, prefix: str) -> np.ndarray:
        tensors = {}
        for fname in PT_FILES:
            data = self._s3_download_bytes(s3, bucket, prefix + fname)
            tensors[fname] = torch.load(io.BytesIO(data), map_location="cpu")

        state_action = torch.cat(
            [
                tensors["eef_torso.pt"],
                tensors["head.pt"],
                tensors["eef_left.pt"],
                tensors["gripper_cmd.pt"][:, 0:1],
                tensors["eef_right.pt"],
                tensors["gripper_cmd.pt"][:, 1:2],
            ],
            dim=-1,
        )
        return state_action.numpy()

    def _extract_video_frames(self, s3, bucket: str, video_key: str) -> list[np.ndarray]:
        data = self._s3_download_bytes(s3, bucket, video_key)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(data)
        tmp.close()
        try:
            frames = self._decode_video(Path(tmp.name))
        finally:
            os.unlink(tmp.name)
        return frames

    def _decode_video(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        target_w, target_h = self._config.image_size
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _s3_download_bytes(s3, bucket: str, key: str) -> bytes:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        path = uri.split("://", 1)[1]
        bucket, _, key = path.partition("/")
        return bucket, key
