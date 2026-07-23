# ruff: noqa: SLF001

from collections import OrderedDict
import io
import json

import numpy as np
import pytest
import torch

from openpi.datasets.ario_dataset import ACTION_DIM
from openpi.datasets.ario_dataset import ArioConfig
from openpi.datasets.ario_dataset import ArioStreamingDataset


def _serialize(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _state_dict(num_frames: int = 3) -> dict[str, torch.Tensor]:
    return {
        "endpose_torso": torch.full((num_frames, 9), 1.0),
        "qpos_head": torch.full((num_frames, 2), 2.0),
        "endpose_left": torch.full((num_frames, 9), 3.0),
        "gripper_left": torch.full((num_frames, 1), 4.0),
        "endpose_right": torch.full((num_frames, 9), 5.0),
        "gripper_right": torch.full((num_frames, 1), 6.0),
    }


def test_deserialize_state_dict_uses_all_31_dimensions():
    state = _state_dict()

    result = ArioStreamingDataset._deserialize_state_action(_serialize(state))

    expected = torch.cat(list(state.values()), dim=-1).numpy()
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, ACTION_DIM)
    assert result.dtype == np.float32


def test_deserialize_state_tensor_is_compatible():
    state = torch.arange(2 * ACTION_DIM, dtype=torch.float32).reshape(2, ACTION_DIM)

    result = ArioStreamingDataset._deserialize_state_action(_serialize(state))

    np.testing.assert_array_equal(result, state.numpy())


def test_deserialize_state_dict_rejects_missing_key():
    state = _state_dict()
    del state["qpos_head"]

    with pytest.raises(KeyError, match="qpos_head"):
        ArioStreamingDataset._deserialize_state_action(_serialize(state))


def test_deserialize_state_dict_rejects_invalid_shape():
    state = _state_dict()
    state["endpose_torso"] = torch.zeros((3, 8))

    with pytest.raises(ValueError, match="endpose_torso"):
        ArioStreamingDataset._deserialize_state_action(_serialize(state))


def test_coarse_instruction_selection_and_nearest_tail():
    payload = {
        "instruction_qwen37_plus_coarse": [
            {
                "instruction": "桌面物品分拣与放置",
                "start_frame": 0,
                "end_frame": 222,
            }
        ]
    }
    segments = ArioStreamingDataset._deserialize_instructions(
        json.dumps(payload, ensure_ascii=False).encode(),
        "instruction_qwen37_plus_coarse",
    )

    assert ArioStreamingDataset._select_instruction(segments, 120, "fallback") == "桌面物品分拣与放置"
    assert ArioStreamingDataset._select_instruction(segments, 235, "fallback") == "桌面物品分拣与放置"


def test_instruction_selection_uses_matching_segment():
    segments = [(0, 20, "任务一"), (21, 40, "任务二")]

    assert ArioStreamingDataset._select_instruction(segments, 6, "fallback") == "任务一"
    assert ArioStreamingDataset._select_instruction(segments, 24, "fallback") == "任务二"


def test_instruction_selection_falls_back_without_segments():
    assert ArioStreamingDataset._select_instruction([], 0, "桌面物品分拣与放置") == "桌面物品分拣与放置"


def test_deserialize_instructions_rejects_missing_key():
    with pytest.raises(ValueError, match="instruction_qwen37_plus_coarse"):
        ArioStreamingDataset._deserialize_instructions(
            json.dumps({"sub_instructions": []}).encode(),
            "instruction_qwen37_plus_coarse",
        )


def test_get_prompt_falls_back_when_instruction_file_is_missing(capsys):
    dataset = object.__new__(ArioStreamingDataset)
    dataset._config = ArioConfig(task="桌面物品分拣与放置", instruction_key="instruction_qwen37_plus_coarse")
    dataset._instruction_cache = OrderedDict()
    dataset._cache_size = 2

    def get_s3():
        return object()

    def raise_missing_file(*_args):
        raise FileNotFoundError("instructions.json")

    dataset._get_s3 = get_s3
    dataset._s3_download_bytes = raise_missing_file

    assert dataset._get_prompt("bucket", "episode/", 0) == "桌面物品分拣与放置"
    assert "using fallback prompt" in capsys.readouterr().out
