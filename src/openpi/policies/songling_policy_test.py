import numpy as np

from openpi.models import model as _model
from openpi.policies import songling_policy


def test_arm_joints_deg_to_rad_converts_joints_and_leaves_grippers():
    qpos = np.array(
        [180.0, 90.0, 0.0, -90.0, 45.0, -180.0, 80.0, 180.0, 90.0, 0.0, -90.0, 45.0, -180.0, 12.5],
        dtype=np.float32,
    )

    converted = songling_policy.arm_joints_deg_to_rad(qpos)

    expected_joints = np.deg2rad([180.0, 90.0, 0.0, -90.0, 45.0, -180.0])
    np.testing.assert_allclose(converted[:6], expected_joints)
    np.testing.assert_allclose(converted[7:13], expected_joints)
    np.testing.assert_allclose(converted[6], 80.0)
    np.testing.assert_allclose(converted[13], 12.5)
    np.testing.assert_array_equal(qpos[6], 80.0)


def test_songling_inputs_maps_three_cameras_and_actions():
    transform = songling_policy.SonglingInputs(model_type=_model.ModelType.PI05)
    example = songling_policy.make_songling_example()
    example["actions"] = np.ones((50, 14), dtype=np.float32)

    result = transform(example)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (50, 14)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(result["image_mask"].values())


def test_songling_outputs_drops_padded_dimensions():
    actions = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)

    result = songling_policy.SonglingOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :14])
