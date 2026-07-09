"""
Convert Xingchen (星尘 Astribot-S1) Ario-format data to LeRobot format for pi0.5 training.

Supports both local directories and direct S3/OSS reading (no full download needed).

Each Ario episode directory contains:
    video.mp4           -- RGB video (384x320 or higher)
    eef_torso.pt        -- [T, 9] torso endpose (xyz3 + rot6d6)
    eef_left.pt         -- [T, 9] left arm endpose (xyz3 + rot6d6)
    eef_right.pt        -- [T, 9] right arm endpose (xyz3 + rot6d6)
    head.pt             -- [T, 2] head joint angles
    gripper_cmd.pt      -- [T, 2] gripper commands (left, right)

The 31-D action/state vector is constructed as:
    endpose_torso(9) + qpos_head(2) + endpose_left(9) + gripper_left(1)
        + endpose_right(9) + gripper_right(1)

Usage (local):
    python examples/xingchen/convert_ario_to_lerobot.py --data_dir /path/to/episodes

Usage (S3/OSS):
    python examples/xingchen/convert_ario_to_lerobot.py \
        --s3_prefixes 's3://bucket/path/prefix1/,s3://bucket/path/prefix2/' \
        --s3_endpoint https://oss-cn-wulanchabu-internal.aliyuncs.com
"""

import io
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm
import tyro

REPO_NAME = "xingchen/fold_clothes"
FPS = 30
ACTION_DIM = 31
IMAGE_SIZE = (320, 240)
PT_FILES = ["eef_torso.pt", "head.pt", "eef_left.pt", "gripper_cmd.pt", "eef_right.pt"]


def _get_s3_client(endpoint: str | None = None):
    import boto3
    from botocore.config import Config
    kwargs = {}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    cfg = Config(s3={"addressing_style": "virtual"}, signature_version="s3v4")
    return boto3.client(
        "s3",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="cn-wulanchabu",
        config=cfg,
        **kwargs,
    )


def _s3_download_bytes(s3, bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key or s3+write://bucket/key -> (bucket, key)."""
    path = uri.split("://", 1)[1]
    bucket, _, key = path.partition("/")
    return bucket, key


def list_s3_episodes(s3, bucket: str, prefix: str) -> list[str]:
    """List episode 'directories' under prefix that contain video.mp4."""
    paginator = s3.get_paginator("list_objects_v2")
    episodes = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/video.mp4"):
                ep_prefix = key[: key.rfind("/video.mp4") + 1]
                episodes.add(ep_prefix)
    return sorted(episodes)


def build_state_action_from_s3(s3, bucket: str, episode_prefix: str) -> np.ndarray:
    """Load .pt files from S3 and concatenate into [T, 31] array."""
    tensors = {}
    for fname in PT_FILES:
        data = _s3_download_bytes(s3, bucket, episode_prefix + fname)
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
    assert state_action.shape[-1] == ACTION_DIM
    return state_action.numpy().astype(np.float32)


def extract_frames_from_s3_video(s3, bucket: str, video_key: str) -> list[np.ndarray]:
    """Download video to temp file, extract frames, delete temp file."""
    data = _s3_download_bytes(s3, bucket, video_key)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data)
    tmp.close()
    try:
        frames = extract_frames_from_video(Path(tmp.name))
    finally:
        os.unlink(tmp.name)
    return frames


def build_state_action_vector(episode_dir: Path) -> np.ndarray:
    """Load .pt files from local dir and concatenate into [T, 31] array."""
    eef_torso = torch.load(episode_dir / "eef_torso.pt", map_location="cpu")
    head = torch.load(episode_dir / "head.pt", map_location="cpu")
    eef_left = torch.load(episode_dir / "eef_left.pt", map_location="cpu")
    gripper_cmd = torch.load(episode_dir / "gripper_cmd.pt", map_location="cpu")
    eef_right = torch.load(episode_dir / "eef_right.pt", map_location="cpu")

    state_action = torch.cat(
        [eef_torso, head, eef_left, gripper_cmd[:, 0:1], eef_right, gripper_cmd[:, 1:2]],
        dim=-1,
    )
    assert state_action.shape[-1] == ACTION_DIM
    return state_action.numpy().astype(np.float32)


def extract_frames_from_video(video_path: Path) -> list[np.ndarray]:
    """Extract all frames from an MP4 file. Returns list of uint8 RGB arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        target_w, target_h = IMAGE_SIZE
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def find_episode_dirs(data_dir: Path) -> list[Path]:
    """Find all valid episode directories (must contain video.mp4 and eef_torso.pt)."""
    episodes = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "video.mp4").exists() and (p / "eef_torso.pt").exists():
            episodes.append(p)
    if (data_dir / "video.mp4").exists() and (data_dir / "eef_torso.pt").exists():
        episodes = [data_dir]
    return episodes


def main(
    data_dir: str = "",
    *,
    s3_prefixes: str = "",
    s3_endpoint: str = "",
    repo_name: str = REPO_NAME,
    fps: int = FPS,
    image_size: tuple[int, int] = IMAGE_SIZE,
    push_to_hub: bool = False,
    video_downsample_rate: int = 1,
    min_frames: int = 0,
):
    """
    Convert Ario episodes to LeRobot format.

    Args:
        data_dir: Path to local directory containing episode subdirectories.
        s3_prefixes: Comma-separated S3 URIs (e.g. 's3://bucket/prefix1/,s3://bucket/prefix2/').
            If provided, reads directly from OSS instead of local disk.
        s3_endpoint: S3-compatible endpoint URL for OSS access.
        repo_name: Output dataset name.
        fps: Recording frequency of the data.
        image_size: Target (width, height) for stored images.
        push_to_hub: Whether to push to HuggingFace Hub.
        video_downsample_rate: Temporal downsampling factor.
        min_frames: Skip episodes shorter than this (before downsampling).
    """
    global IMAGE_SIZE
    IMAGE_SIZE = image_size
    effective_fps = fps // video_downsample_rate

    # Discover episodes
    use_s3 = bool(s3_prefixes)
    s3 = None
    episode_list: list = []

    if use_s3:
        s3 = _get_s3_client(s3_endpoint or None)
        for uri in s3_prefixes.split(","):
            uri = uri.strip()
            if not uri:
                continue
            bucket, prefix = _parse_s3_uri(uri)
            eps = list_s3_episodes(s3, bucket, prefix)
            episode_list.extend([(bucket, ep) for ep in eps])
        if not episode_list:
            raise FileNotFoundError(f"No episodes found in S3 prefixes: {s3_prefixes}")
        print(f"Found {len(episode_list)} episodes on S3")
    else:
        if not data_dir:
            raise ValueError("Must provide either --data_dir or --s3_prefixes")
        data_path = Path(data_dir)
        episode_list = find_episode_dirs(data_path)
        if not episode_list:
            episode_list = sorted(
                p.parent for p in data_path.rglob("video.mp4") if (p.parent / "eef_torso.pt").exists()
            )
        if not episode_list:
            raise FileNotFoundError(f"No valid episodes found in {data_dir}")
        print(f"Found {len(episode_list)} episodes")

    output_path = HF_LEROBOT_HOME / repo_name
    target_w, target_h = image_size

    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="astribot_s1",
        fps=effective_fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (target_h, target_w, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    skipped = 0
    for ep in tqdm(episode_list, desc="Converting episodes"):
        try:
            if use_s3:
                bucket, ep_prefix = ep
                ep_name = ep_prefix.rstrip("/").split("/")[-1]
                state_action = build_state_action_from_s3(s3, bucket, ep_prefix)
                frames = extract_frames_from_s3_video(s3, bucket, ep_prefix + "video.mp4")
            else:
                episode_dir = ep
                ep_name = episode_dir.name
                state_action = build_state_action_vector(episode_dir)
                frames = extract_frames_from_video(episode_dir / "video.mp4")

            num_frames = min(len(frames), len(state_action))
            if num_frames < max(2, min_frames):
                print(f"Skipping {ep_name}: too short ({num_frames} frames)")
                skipped += 1
                continue

            indices = list(range(0, num_frames, video_downsample_rate))
            for i in indices:
                dataset.add_frame(
                    {
                        "image": frames[i],
                        "state": state_action[i],
                        "actions": state_action[i],
                        "task": "fold clothes",
                    }
                )
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {ep_name}: {e}")
            skipped += 1
            continue

    total = len(episode_list)
    print(f"Conversion complete. {total - skipped} episodes converted, {skipped} skipped.")
    print(f"Dataset saved to: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["xingchen", "astribot", "fold_clothes"], private=True, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
