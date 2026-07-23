"""Ario streaming dataset that reads directly from OSS without pre-conversion."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ACTION_DIM = 31
PT_FILES = ["eef_torso.pt", "head.pt", "eef_left.pt", "gripper_cmd.pt", "eef_right.pt"]
IMAGE_SIZE = (320, 240)

# New data format: a single state.pt. It may be either a [T, 31] tensor or a
# dict containing left/right end poses and gripper values.
STATE_PT_FILE = "state.pt"
STATE_DICT_KEYS = ("endpose_left", "endpose_right", "gripper_left", "gripper_right")
INDEX_CACHE_VERSION = 1
INDEX_CACHE_WAIT_SECONDS = 30 * 60


@dataclass
class ArioConfig:
    s3_prefixes: str = ""
    s3_endpoint: str = "https://oss-cn-wulanchabu-internal.aliyuncs.com"
    video_downsample_rate: int = 6
    min_frames: int = 1885
    image_size: tuple[int, int] = IMAGE_SIZE
    task: str = "fold clothes"
    cache_size: int = 32
    max_episodes: int | None = None
    disk_cache_dir: str = "/tmp/ario_disk_cache"
    disk_cache_max_gb: float = 200.0
    skip_video: bool = False


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
        if config.max_episodes is not None:
            episodes = episodes[: config.max_episodes]
        self._episodes = episodes  # list of (bucket, prefix)

        # Build frame index: for each episode, count usable (downsampled) frames.
        # We need to download .pt to know the length, so do a lightweight pass.
        self._episode_lengths: list[int] = []  # downsampled frame count per episode
        self._cumulative: list[int] = []  # cumulative sum for global index lookup
        self._build_index()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        state["_pid"] = None
        state["_cache"] = OrderedDict()
        return state

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
        if self._config.max_episodes is not None:
            episodes = episodes[: self._config.max_episodes]
        print(f"ArioStreamingDataset: found {len(episodes)} episodes on S3")
        return episodes

    def _build_index(self):
        """Load a cached frame index or build it from episode state.pt files."""
        from tqdm import tqdm

        discovery_hash = self._episode_list_hash(self._episodes)
        cache_path = self._index_cache_path()
        if self._load_index_cache(cache_path, discovery_hash):
            return

        # Under torchrun only local rank 0 builds the expensive OSS index. Other
        # local ranks wait for its atomic cache write so they enter DDP together.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if world_size > 1 and local_rank != 0:
            print(f"ArioStreamingDataset: local rank {local_rank} waiting for frame index cache: {cache_path}")
            deadline = time.monotonic() + INDEX_CACHE_WAIT_SECONDS
            while time.monotonic() < deadline:
                if self._load_index_cache(cache_path, discovery_hash):
                    return
                time.sleep(2)
            raise TimeoutError(
                f"Timed out after {INDEX_CACHE_WAIT_SECONDS}s waiting for local rank 0 to build frame index cache: "
                f"{cache_path}"
            )

        s3 = self._get_s3()
        cumulative = 0
        valid_episodes = []
        error_counts: Counter[str] = Counter()
        error_examples: dict[str, tuple[str, str]] = {}
        too_short_count = 0
        too_short_min: int | None = None
        too_short_max: int | None = None
        rate = self._config.video_downsample_rate

        for bucket, prefix in tqdm(self._episodes, desc="Building frame index"):
            try:
                data = self._s3_download_bytes(s3, bucket, prefix + STATE_PT_FILE)
                state_action = self._deserialize_state_action(data)
                raw_len = state_action.shape[0]
            except Exception as exc:
                error_type = type(exc).__name__
                error_counts[error_type] += 1
                state_uri = f"s3://{bucket}/{prefix}{STATE_PT_FILE}"
                error_examples.setdefault(error_type, (state_uri, str(exc)))
                continue

            if raw_len < self._config.min_frames:
                too_short_count += 1
                too_short_min = raw_len if too_short_min is None else min(too_short_min, raw_len)
                too_short_max = raw_len if too_short_max is None else max(too_short_max, raw_len)
                continue

            n_frames = len(range(0, raw_len, rate))
            valid_episodes.append((bucket, prefix))
            self._episode_lengths.append(n_frames)
            cumulative += n_frames
            self._cumulative.append(cumulative)

        self._episodes = valid_episodes
        self._write_index_cache(cache_path, discovery_hash)
        print(f"ArioStreamingDataset: {len(self._episodes)} valid episodes, {cumulative} total frames")
        if error_counts:
            print(
                "ArioStreamingDataset: state.pt load failures: "
                + ", ".join(f"{name}={count}" for name, count in error_counts.most_common())
            )
            for error_type, (state_uri, message) in error_examples.items():
                print(f"  example {error_type}: {state_uri}: {message}")
        if too_short_count:
            print(
                f"ArioStreamingDataset: {too_short_count} episodes shorter than min_frames="
                f"{self._config.min_frames} (observed raw frame range: {too_short_min}-{too_short_max})"
            )

    def _index_cache_path(self) -> Path:
        cache_identity = json.dumps(
            {
                "version": INDEX_CACHE_VERSION,
                "s3_prefixes": self._config.s3_prefixes,
                "s3_endpoint": self._config.s3_endpoint,
                "video_downsample_rate": self._config.video_downsample_rate,
                "min_frames": self._config.min_frames,
                "max_episodes": self._config.max_episodes,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(cache_identity.encode()).hexdigest()[:16]
        return Path(self._config.disk_cache_dir) / f"frame_index_{digest}.json"

    @staticmethod
    def _episode_list_hash(episodes: list[tuple[str, str]]) -> str:
        digest = hashlib.sha256()
        for bucket, prefix in episodes:
            digest.update(bucket.encode())
            digest.update(b"\0")
            digest.update(prefix.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def _load_index_cache(self, path: Path, discovery_hash: str) -> bool:
        try:
            with path.open() as cache_file:
                cached = json.load(cache_file)
            if not isinstance(cached, dict):
                raise ValueError("Frame index cache root must be an object")
            if cached.get("version") != INDEX_CACHE_VERSION:
                return False
            if cached.get("discovery_hash") != discovery_hash:
                return False

            entries = cached["episodes"]
            episodes: list[tuple[str, str]] = []
            lengths: list[int] = []
            for bucket, prefix, length in entries:
                if not isinstance(bucket, str) or not isinstance(prefix, str) or not isinstance(length, int):
                    raise ValueError("Invalid frame index cache entry")
                if length <= 0:
                    raise ValueError(f"Invalid cached episode length: {length}")
                episodes.append((bucket, prefix))
                lengths.append(length)
        except FileNotFoundError:
            return False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"ArioStreamingDataset: ignoring invalid frame index cache {path}: {exc}")
            return False

        cumulative = 0
        self._episodes = episodes
        self._episode_lengths = lengths
        self._cumulative = []
        for length in lengths:
            cumulative += length
            self._cumulative.append(cumulative)
        print(
            f"ArioStreamingDataset: loaded frame index cache with {len(episodes)} valid episodes, "
            f"{cumulative} total frames"
        )
        return True

    def _write_index_cache(self, path: Path, discovery_hash: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_CACHE_VERSION,
            "discovery_hash": discovery_hash,
            "episodes": [
                [bucket, prefix, length]
                for (bucket, prefix), length in zip(self._episodes, self._episode_lengths, strict=True)
            ],
        }
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temp_path.open("w") as cache_file:
                json.dump(payload, cache_file, separators=(",", ":"))
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"ArioStreamingDataset: wrote frame index cache: {path}")

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int, _retries: int = 3, _timeout: float = 60.0) -> dict:
        import random
        import signal

        if index < 0:
            index += len(self)

        for attempt in range(_retries):
            ep_idx, frame_idx = self._global_to_local(index)
            bucket, prefix = self._episodes[ep_idx]
            cache_key = prefix

            try:
                if self._config.skip_video:
                    state_action = self._get_episode_state(bucket, prefix, cache_key, _timeout)
                    h, w = self._config.image_size[1], self._config.image_size[0]
                    image = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    state_action, frames = self._get_episode_with_timeout(bucket, prefix, cache_key, _timeout)
                    image = frames[frame_idx]

                state = state_action[frame_idx]
                actions = self._get_action_chunk(state_action, frame_idx)

                return {
                    "observation/image": image,
                    "observation/state": state,
                    "actions": actions,
                    "prompt": self._config.task,
                }
            except (TimeoutError, Exception) as e:
                print(f"[WARN] Episode fetch failed (attempt {attempt+1}/{_retries}), ep={ep_idx}: {e}. Skipping.", flush=True)
                index = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Failed to fetch any episode after {_retries} retries")

    def _get_episode_with_timeout(self, bucket, prefix, cache_key, timeout):
        """Wrap _get_episode with a timeout. Falls back to no-timeout on non-main threads."""
        import signal
        import threading

        if threading.current_thread() is not threading.main_thread():
            return self._get_episode(bucket, prefix, cache_key)

        def _handler(signum, frame):
            raise TimeoutError(f"Episode fetch timed out after {timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
        try:
            result = self._get_episode(bucket, prefix, cache_key)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

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

    def _get_episode_state(
        self, bucket: str, prefix: str, cache_key: str, timeout: float
    ) -> np.ndarray:
        """Get only state/action data (no video). Used when skip_video=True."""
        state_key = cache_key + "__state"
        if state_key in self._cache:
            self._cache.move_to_end(state_key)
            return self._cache[state_key]

        s3 = self._get_s3()
        state_action = self._build_state_action(s3, bucket, prefix)

        rate = self._config.video_downsample_rate
        raw_len = len(state_action)
        indices = list(range(0, raw_len, rate))
        state_action = state_action[indices].astype(np.float32)

        self._cache[state_key] = state_action
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return state_action

    def _get_episode(
        self, bucket: str, prefix: str, cache_key: str
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get episode data, using memory LRU cache backed by disk cache."""
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Try disk cache
        disk_path = self._disk_cache_path(cache_key)
        if disk_path.exists():
            try:
                data = np.load(disk_path, allow_pickle=True)
                state_action = data["state_action"]
                frames = list(data["frames"])
                disk_path.stat()  # touch atime for LRU
                os.utime(disk_path, None)
                self._cache[cache_key] = (state_action, frames)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return state_action, frames
            except Exception:
                disk_path.unlink(missing_ok=True)

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

        # Save to disk cache
        self._save_to_disk_cache(disk_path, state_action, frames)

        self._cache[cache_key] = (state_action, frames)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return state_action, frames

    def _disk_cache_path(self, cache_key: str) -> Path:
        import hashlib
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = Path(self._config.disk_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key_hash}.npz"

    def _save_to_disk_cache(self, path: Path, state_action: np.ndarray, frames: list[np.ndarray]):
        try:
            self._evict_disk_cache_if_needed()
            frames_arr = np.stack(frames)
            np.savez(path, state_action=state_action, frames=frames_arr)
        except Exception:
            path.unlink(missing_ok=True)

    def _evict_disk_cache_if_needed(self):
        cache_dir = Path(self._config.disk_cache_dir)
        if not cache_dir.exists():
            return
        max_bytes = int(self._config.disk_cache_max_gb * 1024**3)
        files = list(cache_dir.glob("*.npz"))
        total = sum(f.stat().st_size for f in files)
        if total <= max_bytes:
            return
        files.sort(key=lambda f: f.stat().st_atime)
        for f in files:
            if total <= max_bytes * 0.8:
                break
            total -= f.stat().st_size
            f.unlink(missing_ok=True)

    def _build_state_action(self, s3, bucket: str, prefix: str) -> np.ndarray:
        data = self._s3_download_bytes(s3, bucket, prefix + STATE_PT_FILE)
        return self._deserialize_state_action(data)

    @staticmethod
    def _deserialize_state_action(data: bytes) -> np.ndarray:
        """Convert supported state.pt formats to the model's [T, 31] layout."""
        state = torch.load(io.BytesIO(data), map_location="cpu")

        if isinstance(state, torch.Tensor):
            if state.ndim != 2 or state.shape[1] != ACTION_DIM:
                raise ValueError(f"Expected state tensor with shape [T, {ACTION_DIM}], got {tuple(state.shape)}")
            return state.detach().cpu().numpy().astype(np.float32)

        if not isinstance(state, dict):
            raise TypeError(f"Expected state.pt to contain a tensor or dict, got {type(state).__name__}")

        missing_keys = [key for key in STATE_DICT_KEYS if key not in state]
        if missing_keys:
            raise KeyError(f"state.pt is missing required keys: {missing_keys}")

        tensors = [state[key] for key in STATE_DICT_KEYS]
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            value_types = {key: type(state[key]).__name__ for key in STATE_DICT_KEYS}
            raise TypeError(f"State dict values must be tensors, got {value_types}")

        endpose_left, endpose_right, gripper_left, gripper_right = tensors
        expected_widths = {
            "endpose_left": 9,
            "endpose_right": 9,
            "gripper_left": 1,
            "gripper_right": 1,
        }
        lengths = {value.shape[0] for value in tensors if value.ndim == 2}
        invalid_shapes = {
            key: tuple(state[key].shape)
            for key, width in expected_widths.items()
            if state[key].ndim != 2 or state[key].shape[1] != width
        }
        if invalid_shapes:
            raise ValueError(f"Unexpected state dict tensor shapes: {invalid_shapes}")
        if len(lengths) != 1:
            raise ValueError(f"State dict tensors have inconsistent lengths: {[tuple(value.shape) for value in tensors]}")

        # Existing Xingchen layout:
        # torso(9) + head(2) + left(9) + left gripper(1) + right(9) + right gripper(1).
        # The new dict format has no torso/head values, so those 11 dimensions are fixed at zero.
        num_frames = endpose_left.shape[0]
        torso_and_head = torch.zeros((num_frames, 11), dtype=endpose_left.dtype)
        state_action = torch.cat(
            [torso_and_head, endpose_left, gripper_left, endpose_right, gripper_right],
            dim=-1,
        )
        return state_action.detach().cpu().numpy().astype(np.float32)

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
