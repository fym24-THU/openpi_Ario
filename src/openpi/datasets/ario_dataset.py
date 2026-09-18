"""Ario streaming dataset that reads directly from OSS without pre-conversion."""

from __future__ import annotations

from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from decord import VideoReader
from decord import cpu
import filelock
import numpy as np
import torch

XINGCHEN_ACTION_DIM = 31
XINGCHEN_PT_FILES = ["eef_torso.pt", "head.pt", "eef_left.pt", "gripper_cmd.pt", "eef_right.pt"]
SONGLING_QPOS14_INDICES = (0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 33)
SONGLING_QPOS_PADDING_INDICES = (6, 23)
IMAGE_SIZE = (320, 240)

CAMERA_VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DISK_CACHE_FORMAT_VERSION = 5
PROXY_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
DEFAULT_PG_HOST = "10.107.2.51"
DEFAULT_PG_PORT = 55434
DEFAULT_PG_DATABASE = "embodied"
DEFAULT_PG_USER = "ssemb_write"
EPISODE_SELECTION_SQL = """
SELECT DISTINCT episode_path
FROM embodied.ario_format_data
WHERE episode_path LIKE %s
  AND ((state >> 0) & 3) = 1
  AND ((state >> 2) & 3) != 2
  AND ((state >> 4) & 3) != 2
ORDER BY episode_path
""".strip()


@dataclass
class ArioConfig:
    s3_prefixes: str = ""
    s3_endpoint: str = "https://oss-cn-wulanchabu-internal.aliyuncs.com"
    excluded_episodes: tuple[str, ...] = ()
    video_downsample_rate: int = 1
    min_frames: int = 1885
    image_size: tuple[int, int] = IMAGE_SIZE
    task: str = "fold clothes"
    load_instructions: bool = False
    instruction_field: str = "sub_instructions"
    skip_video: bool = False
    cache_size: int = 32
    video_reader_cache_size: int = 16
    episode_frames_per_batch: int = 0
    max_episodes: int | None = None
    disk_cache_dir: str = "/tmp/ario_disk_cache"
    disk_cache_max_gb: float = 200.0
    multi_view: bool = True
    data_format: str = "xingchen"
    action_start_offset: int = 0
    filter_episodes_by_state: bool = False


@contextlib.contextmanager
def _without_proxies():
    previous = {name: os.environ.pop(name) for name in PROXY_ENV_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


def _data_lake_password() -> str:
    password = os.environ.get("SSEMB_PG_PASSWORD")
    if password:
        return password
    password_file = Path(
        os.environ.get(
            "SSEMB_PG_PASSWORD_FILE",
            "/tmp/ssemb-datalake-postgres-secrets/read_password",
        )
    )
    if password_file.is_file():
        return password_file.read_text(encoding="utf-8").strip()
    return "embodied-write"


class ArioStreamingDataset:
    """Dataset that streams Ario-format episodes directly from S3/OSS.

    Each __getitem__ returns a single training sample with action chunking applied.
    Episodes are cached in an LRU manner to avoid redundant downloads.
    """

    def __init__(self, config: ArioConfig, action_horizon: int):
        if config.data_format not in {"xingchen", "songling_canonical55"}:
            raise ValueError(f"Unsupported Ario data format: {config.data_format}")
        if config.action_start_offset < 0:
            raise ValueError("action_start_offset must be non-negative")
        if config.video_reader_cache_size < 0:
            raise ValueError("video_reader_cache_size must be non-negative")
        if config.episode_frames_per_batch < 0:
            raise ValueError("episode_frames_per_batch must be non-negative")

        self._config = config
        self._action_horizon = action_horizon
        self._s3 = None
        self._pid: int | None = None

        # Per-process LRU for small state/action arrays. Videos remain as raw
        # MP4 files in the shared object cache and are decoded sparsely.
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = config.cache_size
        self._video_readers: OrderedDict[Path, VideoReader] = OrderedDict()

        # Per-episode frame ranges loaded from instructions.json.
        self._instructions: dict[str, list[tuple[int, int, str]]] = {}

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
        state["_video_readers"] = OrderedDict()
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
            self._video_readers.clear()
        return self._s3

    def _discover_episodes(self) -> list[tuple[str, str]]:
        s3 = self._get_s3()
        episodes: set[tuple[str, str]] = set()
        selected_episodes = self._select_episodes_from_data_lake() if self._config.filter_episodes_by_state else None
        for uri in self._config.s3_prefixes.split(","):
            uri = uri.strip()
            if not uri:
                continue
            bucket, prefix = self._parse_s3_uri(uri)
            paginator = s3.get_paginator("list_objects_v2")
            discovered_views: dict[str, set[str]] = {}
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if self._config.multi_view:
                        for cam in CAMERA_VIEWS:
                            suffix = f"raw_video/{cam}.mp4"
                            if key.endswith(suffix):
                                ep_prefix = key[: -len(suffix)]
                                if selected_episodes is not None and (bucket, ep_prefix) not in selected_episodes:
                                    break
                                discovered_views.setdefault(ep_prefix, set()).add(cam)
                                break
                    elif key.endswith("/video.mp4"):
                        ep_prefix = key[: key.rfind("/video.mp4") + 1]
                        episode = (bucket, ep_prefix)
                        if selected_episodes is None or episode in selected_episodes:
                            episodes.add(episode)

            if self._config.multi_view:
                required_views = set(CAMERA_VIEWS)
                for ep_prefix, views in discovered_views.items():
                    if views == required_views:
                        episodes.add((bucket, ep_prefix))
                    else:
                        missing = sorted(required_views - views)
                        print(
                            f"[WARN] Skipping incomplete multi-view episode {ep_prefix}: "
                            f"missing {missing}",
                            flush=True,
                        )

        excluded_episodes = {
            (bucket, prefix.rstrip("/") + "/")
            for uri in self._config.excluded_episodes
            for bucket, prefix in [self._parse_s3_uri(uri)]
        }
        excluded_count = len(episodes & excluded_episodes)
        episodes.difference_update(excluded_episodes)
        if excluded_count:
            print(f"ArioStreamingDataset: excluded {excluded_count} configured episodes")

        episodes = sorted(episodes, key=lambda x: (x[0], x[1]))
        if self._config.max_episodes is not None:
            episodes = episodes[: self._config.max_episodes]
        print(f"ArioStreamingDataset: found {len(episodes)} episodes on S3")
        return episodes

    def _select_episodes_from_data_lake(self) -> set[tuple[str, str]]:
        """Select successful episode paths before scanning their object-store files."""
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "Filtering Ario episodes through the data lake requires the psycopg dependency"
            ) from e

        connect_kwargs = {
            "host": os.environ.get("SSEMB_PG_HOST", DEFAULT_PG_HOST),
            "port": int(os.environ.get("SSEMB_PG_PORT", str(DEFAULT_PG_PORT))),
            "dbname": os.environ.get("SSEMB_PG_DATABASE", DEFAULT_PG_DATABASE),
            "user": os.environ.get("SSEMB_PG_USER", DEFAULT_PG_USER),
            "password": _data_lake_password(),
            "connect_timeout": int(os.environ.get("SSEMB_PG_CONNECT_TIMEOUT", "30")),
        }
        selected: set[tuple[str, str]] = set()
        with (
            _without_proxies(),
            psycopg.connect(**connect_kwargs) as connection,
            connection.cursor() as cursor,
        ):
            for uri in self._config.s3_prefixes.split(","):
                uri = uri.strip()
                if not uri:
                    continue
                bucket, prefix = self._parse_s3_uri(uri)
                canonical_prefix = f"s3://{bucket}/{prefix.rstrip('/')}/"
                cursor.execute(EPISODE_SELECTION_SQL, (canonical_prefix + "%",))
                for (episode_path,) in cursor.fetchall():
                    episode_bucket, episode_prefix = self._parse_s3_uri(str(episode_path))
                    if episode_bucket != bucket or not episode_prefix.startswith(prefix):
                        raise RuntimeError(f"Data lake returned episode outside configured prefix: {episode_path}")
                    selected.add((episode_bucket, episode_prefix.rstrip("/") + "/"))

        print(f"ArioStreamingDataset: selected {len(selected)} episodes from data lake")
        return selected

    def _build_index(self):
        """Download the format-specific state file from each episode to determine its length."""
        from tqdm import tqdm

        s3 = self._get_s3()
        cumulative = 0
        valid_episodes = []
        rate = self._config.video_downsample_rate

        for bucket, prefix in tqdm(self._episodes, desc="Building frame index"):
            try:
                if self._config.data_format == "songling_canonical55":
                    state_path = self._get_cached_object(s3, bucket, prefix + "state.pt")
                    state_dict = torch.load(state_path, map_location="cpu")
                    raw_len = state_dict["__canonical55__"].shape[0]
                else:
                    state_path = self._get_cached_object(s3, bucket, prefix + "eef_torso.pt")
                    tensor = torch.load(state_path, map_location="cpu")
                    raw_len = tensor.shape[0]
            except Exception:
                continue

            if raw_len < self._config.min_frames:
                continue

            # Load per-episode instruction from instructions.json
            if self._config.load_instructions:
                try:
                    instruction_path = self._get_cached_object(s3, bucket, prefix + "instructions.json")
                    instr_json = json.loads(instruction_path.read_text(encoding="utf-8"))
                    self._instructions[prefix] = self._parse_instruction_segments(
                        instr_json, self._config.instruction_field
                    )
                except Exception as e:
                    print(
                        f"[WARN] Failed to load instruction field {self._config.instruction_field!r} "
                        f"for {prefix}: {e}",
                        flush=True,
                    )

            n_frames = len(range(0, raw_len, rate))
            valid_episodes.append((bucket, prefix))
            self._episode_lengths.append(n_frames)
            cumulative += n_frames
            self._cumulative.append(cumulative)

        self._episodes = valid_episodes
        print(f"ArioStreamingDataset: {len(self._episodes)} valid episodes, {cumulative} total frames")

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    @property
    def episode_lengths(self) -> tuple[int, ...]:
        """Return the downsampled frame count for each discovered episode."""
        return tuple(self._episode_lengths)

    def global_to_local(self, index: int) -> tuple[int, int]:
        """Convert a global dataset index to an episode index and local frame index."""
        return self._global_to_local(index)

    def get_item(self, index: int, *, retries: int = 3, timeout: float = 60.0) -> dict:
        """Fetch a sample with explicit retry behavior."""
        return self.__getitem__(index, _retries=retries, _timeout=timeout)

    def __getitem__(self, index: int, _retries: int = 3, _timeout: float = 60.0) -> dict:
        import random

        if index < 0:
            index += len(self)

        for attempt in range(_retries):
            ep_idx, frame_idx = self._global_to_local(index)
            bucket, prefix = self._episodes[ep_idx]
            cache_key = f"{bucket}/{prefix}"

            try:
                states, action_source = self._get_episode_with_timeout(
                    bucket, prefix, cache_key, _timeout
                )

                state = states[frame_idx]
                actions = self._get_action_chunk(action_source, frame_idx)
                prompt = self._get_prompt(prefix, frame_idx)
                raw_frame_idx = frame_idx * self._config.video_downsample_rate

                if self._config.skip_video:
                    target_w, target_h = self._config.image_size
                    frames = {
                        camera: np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        for camera in CAMERA_VIEWS
                    }
                elif self._config.multi_view:
                    frames = {
                        camera: self._read_video_frame(
                            s3=self._get_s3(),
                            bucket=bucket,
                            video_key=prefix + f"raw_video/{camera}.mp4",
                            frame_index=raw_frame_idx,
                        )
                        for camera in CAMERA_VIEWS
                    }
                else:
                    frame = self._read_video_frame(
                        s3=self._get_s3(),
                        bucket=bucket,
                        video_key=prefix + "video.mp4",
                        frame_index=raw_frame_idx,
                    )
                    frames = dict.fromkeys(CAMERA_VIEWS, frame)

                result = {
                    "observation/image": frames["cam_high"],
                    "observation/state": state,
                    "actions": actions,
                    "prompt": prompt,
                }

                if self._config.multi_view:
                    for cam in CAMERA_VIEWS:
                        result[f"observation/{cam}"] = frames[cam]

                return result
            except (TimeoutError, Exception) as e:
                print(
                    f"[WARN] Episode fetch failed (attempt {attempt+1}/{_retries}), "
                    f"ep={ep_idx}: {e}. Skipping.",
                    flush=True,
                )
                index = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Failed to fetch any episode after {_retries} retries")

    @staticmethod
    def _parse_instruction_segments(instruction_data: dict, field: str) -> list[tuple[int, int, str]]:
        """Parse inclusive frame ranges from a configured instructions.json field."""
        raw_segments = instruction_data.get(field)
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"Instruction field {field!r} must be a non-empty list")

        segments = []
        for index, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                raise ValueError(f"Instruction segment {index} in {field!r} must be an object")
            try:
                start_frame = int(segment["start_frame"])
                end_frame = int(segment["end_frame"])
                instruction = str(segment["instruction"]).strip()
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"Instruction segment {index} in {field!r} requires instruction, start_frame, and end_frame"
                ) from e
            if start_frame < 0 or end_frame < start_frame:
                raise ValueError(
                    f"Invalid frame range [{start_frame}, {end_frame}] in instruction segment {index}"
                )
            if not instruction:
                raise ValueError(f"Instruction segment {index} in {field!r} has empty instruction")
            segments.append((start_frame, end_frame, instruction))

        return sorted(segments, key=lambda segment: (segment[0], segment[1]))

    def _get_prompt(self, prefix: str, frame_idx: int) -> str:
        """Return the instruction whose inclusive raw-frame range contains this sample."""
        raw_frame_idx = frame_idx * self._config.video_downsample_rate
        for start_frame, end_frame, instruction in self._instructions.get(prefix, ()):
            if start_frame <= raw_frame_idx <= end_frame:
                return instruction
        return self._config.task

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

    def _get_action_chunk(self, action_source: np.ndarray, frame_idx: int) -> np.ndarray:
        """Get an action chunk after the configured offset, clamping at the episode end."""
        n = len(action_source)
        start = frame_idx + self._config.action_start_offset
        indices = [min(start + i, n - 1) for i in range(self._action_horizon)]
        return action_source[indices]

    def _get_episode(
        self, bucket: str, prefix: str, cache_key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and downsample state/action arrays, using a per-process LRU."""
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        states, action_source = self._build_state_and_actions(self._get_s3(), bucket, prefix)
        indices = np.arange(0, min(len(states), len(action_source)), self._config.video_downsample_rate)
        episode = (
            states[indices].astype(np.float32),
            action_source[indices].astype(np.float32),
        )
        return self._remember_episode(cache_key, episode)

    def _remember_episode(
        self,
        cache_key: str,
        episode: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        self._cache[cache_key] = episode
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return episode

    def _object_cache_path(self, bucket: str, key: str) -> Path:
        cache_identity = f"v{DISK_CACHE_FORMAT_VERSION}|s3://{bucket}/{key}"
        key_hash = hashlib.sha256(cache_identity.encode()).hexdigest()
        suffix = Path(key).suffix or ".obj"
        cache_dir = Path(self._config.disk_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key_hash}{suffix}"

    def _get_cached_object(
        self,
        s3,
        bucket: str,
        key: str,
    ) -> Path:
        """Cache one raw OSS object with a process-safe lock and atomic rename."""
        path = self._object_cache_path(bucket, key)
        if path.is_file():
            os.utime(path, None)
            return path

        lock_path = path.with_suffix(f"{path.suffix}.lock")
        with filelock.FileLock(lock_path):
            if path.is_file():
                os.utime(path, None)
                return path

            temporary_path: Path | None = None
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                try:
                    response = s3.get_object(Bucket=bucket, Key=key)
                    with response["Body"] as body:
                        shutil.copyfileobj(body, temporary_file, length=8 * 1024 * 1024)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                except Exception:
                    temporary_path.unlink(missing_ok=True)
                    raise
            os.replace(temporary_path, path)

        self._evict_object_cache(protected_path=path)
        return path

    def _evict_object_cache(self, *, protected_path: Path) -> None:
        cache_dir = Path(self._config.disk_cache_dir)
        if not cache_dir.exists():
            return
        max_bytes = int(self._config.disk_cache_max_gb * 1024**3)
        files = [
            path
            for path in cache_dir.iterdir()
            if path.is_file() and not path.name.endswith((".lock", ".tmp"))
        ]
        total = sum(f.stat().st_size for f in files)
        if total <= max_bytes:
            return
        files.sort(key=lambda f: f.stat().st_atime)
        for path in files:
            if total <= max_bytes * 0.8:
                break
            if path == protected_path:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
            except FileNotFoundError:
                pass

    def _build_state_and_actions(self, s3, bucket: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        if self._config.data_format == "songling_canonical55":
            state_path = self._get_cached_object(s3, bucket, prefix + "state.pt")
            state_dict = torch.load(state_path, map_location="cpu")
            states = self._extract_songling_qpos14(state_dict)
            return states, states

        tensors = {}
        for fname in XINGCHEN_PT_FILES:
            tensor_path = self._get_cached_object(s3, bucket, prefix + fname)
            tensors[fname] = torch.load(tensor_path, map_location="cpu")

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
        values = state_action.numpy()
        if values.shape[-1] != XINGCHEN_ACTION_DIM:
            raise ValueError(f"Expected {XINGCHEN_ACTION_DIM} Xingchen dimensions, got {values.shape[-1]}")
        return values, values

    @staticmethod
    def _extract_songling_qpos14(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
        """Extract 14-D Songling qpos from canonical55 and convert arm joints to radians.

        Layout: [left qpos6, left gripper, right qpos6, right gripper]. Canonical55 stores
        arm joints in degrees; grippers remain in raw encoder units. Xingchen data is not
        converted here.
        """
        canonical = state_dict.get("__canonical55__")
        mask = state_dict.get("__canonical55_mask__")
        if canonical is None or mask is None:
            raise KeyError("Songling state.pt must contain __canonical55__ and __canonical55_mask__")
        if canonical.ndim != 2 or canonical.shape[-1] != 55:
            raise ValueError(f"Expected __canonical55__ shape [T, 55], got {tuple(canonical.shape)}")
        if mask.shape != canonical.shape:
            raise ValueError(
                f"Expected __canonical55_mask__ shape {tuple(canonical.shape)}, got {tuple(mask.shape)}"
            )

        selected_mask = mask[:, SONGLING_QPOS14_INDICES]
        if not torch.all(selected_mask > 0):
            raise ValueError("Songling canonical55 qpos/gripper slots must all be valid")
        padding_mask = mask[:, SONGLING_QPOS_PADDING_INDICES]
        if not torch.all(padding_mask == 0):
            raise ValueError("Songling canonical55 seventh joint slots must be invalid padding")

        from openpi.policies.songling_policy import arm_joints_deg_to_rad

        return arm_joints_deg_to_rad(canonical[:, SONGLING_QPOS14_INDICES].detach().cpu().numpy())

    def _read_video_frame(
        self,
        *,
        s3,
        bucket: str,
        video_key: str,
        frame_index: int,
    ) -> np.ndarray:
        """Decode one indexed RGB frame from a raw MP4 cached on local disk."""
        video_path = self._get_cached_object(s3, bucket, video_key)
        reader = self._get_video_reader(video_path)
        if frame_index < 0 or frame_index >= len(reader):
            raise IndexError(
                f"Frame {frame_index} is outside video {video_key!r} with {len(reader)} frames"
            )
        try:
            return reader.get_batch([frame_index]).asnumpy()[0]
        except Exception:
            self._video_readers.pop(video_path, None)
            raise

    def _get_video_reader(self, video_path: Path) -> VideoReader:
        """Return a process-local cached reader for one video."""
        if not hasattr(self, "_video_readers"):
            self._video_readers = OrderedDict()

        if self._config.video_reader_cache_size == 0:
            return VideoReader(str(video_path), ctx=cpu(0), num_threads=1)

        reader = self._video_readers.pop(video_path, None)
        if reader is not None:
            self._video_readers[video_path] = reader
            return reader

        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        self._video_readers[video_path] = reader
        while len(self._video_readers) > self._config.video_reader_cache_size:
            self._video_readers.popitem(last=False)
        return reader

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        path = uri.split("://", 1)[1]
        bucket, _, key = path.partition("/")
        return bucket, key
