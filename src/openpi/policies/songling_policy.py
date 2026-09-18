"""Policy transforms for Songling dual-arm qpos control.

Ario canonical55 stores Songling arm joints in degrees. Training and this policy
use radians: `ArioStreamingDataset` converts the 12 arm joints on load. Gripper
slots stay in raw encoder units. Xingchen data is already in radians and is not
converted.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 14
# True for the 6+6 arm joints; False for the two gripper slots.
ARM_JOINT_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def arm_joints_deg_to_rad(qpos: np.ndarray) -> np.ndarray:
    """Convert Songling 14-D arm joints from degrees to radians. Grippers unchanged."""
    qpos = np.array(qpos, dtype=np.float32, copy=True)
    if qpos.shape[-1] < ACTION_DIM:
        raise ValueError(f"Expected at least {ACTION_DIM} Songling dimensions, got {qpos.shape}")
    joints = qpos[..., :ACTION_DIM]
    joints[..., ARM_JOINT_MASK] = np.deg2rad(joints[..., ARM_JOINT_MASK])
    qpos[..., :ACTION_DIM] = joints
    return qpos


def make_songling_example() -> dict:
    """Create a random Songling policy input example."""
    return {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_high": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_right_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "prompt": "fold clothes",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SonglingInputs(transforms.DataTransformFn):
    """Map Songling observations to the three image slots expected by pi0."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "observation/cam_high" in data:
            base_image = _parse_image(data["observation/cam_high"])
        else:
            base_image = _parse_image(data["observation/image"])

        if "observation/cam_left_wrist" in data:
            left_wrist_image = _parse_image(data["observation/cam_left_wrist"])
            left_wrist_mask = np.True_
        else:
            left_wrist_image = np.zeros_like(base_image)
            left_wrist_mask = np.False_

        if "observation/cam_right_wrist" in data:
            right_wrist_image = _parse_image(data["observation/cam_right_wrist"])
            right_wrist_mask = np.True_
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.False_

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SonglingOutputs(transforms.DataTransformFn):
    """Return only the 14 physical Songling action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
