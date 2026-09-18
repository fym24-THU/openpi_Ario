"""Tests for Ario multi-view discovery and cache isolation."""

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from openpi.datasets.ario_dataset import EPISODE_SELECTION_SQL
from openpi.datasets.ario_dataset import SONGLING_QPOS14_INDICES
from openpi.datasets.ario_dataset import ArioConfig
from openpi.datasets.ario_dataset import ArioStreamingDataset
from openpi.policies.songling_policy import arm_joints_deg_to_rad


class _FakePaginator:
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, **_kwargs):
        yield {"Contents": [{"Key": key} for key in self._keys]}


class _FakeS3:
    def __init__(self, keys):
        self._keys = keys

    def get_paginator(self, _name):
        return _FakePaginator(self._keys)


class ArioMultiViewDatasetTest(unittest.TestCase):
    def test_discovers_only_complete_raw_multiview_episodes(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(
            s3_prefixes="oss://bucket/root/",
            multi_view=True,
        )
        dataset._get_s3 = lambda: _FakeS3(
            [
                "root/complete/raw_video/cam_high.mp4",
                "root/complete/raw_video/cam_left_wrist.mp4",
                "root/complete/raw_video/cam_right_wrist.mp4",
                "root/incomplete/raw_video/cam_high.mp4",
                "root/incomplete/raw_video/cam_left_wrist.mp4",
            ]
        )

        self.assertEqual(
            dataset._discover_episodes(),
            [("bucket", "root/complete/")],
        )

    def test_excludes_configured_episode_uri(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(
            s3_prefixes="s3://bucket/root/",
            excluded_episodes=("s3://bucket/root/rejected/",),
            multi_view=True,
        )
        dataset._get_s3 = lambda: _FakeS3(
            [
                f"root/{episode}/raw_video/{camera}.mp4"
                for episode in ("kept", "rejected")
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            ]
        )

        self.assertEqual(dataset._discover_episodes(), [("bucket", "root/kept/")])

    def test_data_lake_selection_limits_discovered_episodes(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(
            s3_prefixes="s3://bucket/root/",
            multi_view=True,
            filter_episodes_by_state=True,
        )
        dataset._get_s3 = lambda: _FakeS3(
            [
                f"root/{episode}/raw_video/{camera}.mp4"
                for episode in ("selected", "rejected")
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            ]
        )
        dataset._select_episodes_from_data_lake = lambda: {("bucket", "root/selected/")}

        self.assertEqual(dataset._discover_episodes(), [("bucket", "root/selected/")])

    def test_data_lake_query_contains_required_state_filters(self):
        self.assertIn("((state >> 0) & 3) = 1", EPISODE_SELECTION_SQL)
        self.assertIn("((state >> 2) & 3) != 2", EPISODE_SELECTION_SQL)
        self.assertIn("((state >> 4) & 3) != 2", EPISODE_SELECTION_SQL)

    def test_object_cache_identity_uses_bucket_key_and_file_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
            dataset._config = ArioConfig(disk_cache_dir=tmpdir)
            first = dataset._object_cache_path("bucket-a", "root/video.mp4")
            second = dataset._object_cache_path("bucket-b", "root/video.mp4")
            state = dataset._object_cache_path("bucket-a", "root/state.pt")

            self.assertNotEqual(first, second)
            self.assertEqual(first.suffix, ".mp4")
            self.assertEqual(state.suffix, ".pt")

    def test_failed_object_download_leaves_no_partial_or_final_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
            dataset._config = ArioConfig(disk_cache_dir=tmpdir)

            body = mock.MagicMock()
            body.__enter__.return_value.read.side_effect = OSError("network failure")
            s3 = mock.MagicMock()
            s3.get_object.return_value = {"Body": body}
            final_path = dataset._object_cache_path("bucket", "root/video.mp4")

            with self.assertRaisesRegex(OSError, "network failure"):
                dataset._get_cached_object(s3, "bucket", "root/video.mp4")

            self.assertFalse(final_path.exists())
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

    def test_object_cache_lock_prevents_duplicate_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class SlowS3:
                def __init__(self):
                    self.call_count = 0

                def get_object(self, **_kwargs):
                    self.call_count += 1
                    time.sleep(0.1)
                    return {"Body": io.BytesIO(b"video-bytes")}

            s3 = SlowS3()

            def make_dataset():
                dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
                dataset._config = ArioConfig(disk_cache_dir=tmpdir)
                return dataset

            datasets = (make_dataset(), make_dataset())
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = list(
                    executor.map(
                        lambda dataset: dataset._get_cached_object(
                            s3, "bucket", "root/video.mp4"
                        ),
                        datasets,
                    )
                )

            self.assertEqual(s3.call_count, 1)
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(paths[0].read_bytes(), b"video-bytes")

    def test_sparse_video_reader_decodes_only_requested_frame(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig()
        dataset._get_cached_object = lambda *_args: Path("/tmp/cached.mp4")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        batch = mock.MagicMock()
        batch.asnumpy.return_value = frame[None]
        reader = mock.MagicMock()
        reader.__len__.return_value = 10
        reader.get_batch.return_value = batch

        with mock.patch("openpi.datasets.ario_dataset.VideoReader", return_value=reader):
            result = dataset._read_video_frame(
                s3=None,
                bucket="bucket",
                video_key="root/video.mp4",
                frame_index=7,
            )

        self.assertEqual(result.shape, (480, 640, 3))
        reader.get_batch.assert_called_once_with([7])

    def test_video_reader_cache_reuses_and_evicts_readers(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(video_reader_cache_size=2)
        readers = [mock.MagicMock(name=f"reader_{index}") for index in range(4)]

        with mock.patch("openpi.datasets.ario_dataset.VideoReader", side_effect=readers) as constructor:
            first = dataset._get_video_reader(Path("/tmp/first.mp4"))
            self.assertIs(dataset._get_video_reader(Path("/tmp/first.mp4")), first)
            dataset._get_video_reader(Path("/tmp/second.mp4"))
            dataset._get_video_reader(Path("/tmp/third.mp4"))
            reloaded_first = dataset._get_video_reader(Path("/tmp/first.mp4"))

        self.assertIsNot(reloaded_first, first)
        self.assertEqual(constructor.call_count, 4)

    def test_getitem_sparse_decodes_current_raw_frame_from_all_views(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(video_downsample_rate=3, multi_view=True)
        dataset._action_horizon = 2
        dataset._episodes = [("bucket", "episode/")]
        dataset._cumulative = [4]
        states = np.arange(4 * 14, dtype=np.float32).reshape(4, 14)
        dataset._get_episode_with_timeout = lambda *_args: (states, states)
        dataset._get_prompt = lambda *_args: "fold"
        dataset._get_s3 = lambda: None
        requests = []

        def read_frame(**kwargs):
            requests.append((kwargs["video_key"], kwargs["frame_index"]))
            return np.zeros((4, 5, 3), dtype=np.uint8)

        dataset._read_video_frame = read_frame
        sample = dataset[2]

        self.assertEqual(
            requests,
            [
                ("episode/raw_video/cam_high.mp4", 6),
                ("episode/raw_video/cam_left_wrist.mp4", 6),
                ("episode/raw_video/cam_right_wrist.mp4", 6),
            ],
        )
        self.assertEqual(sample["observation/cam_high"].shape, (4, 5, 3))
        np.testing.assert_array_equal(sample["observation/state"], states[2])

    def test_state_episode_cache_only_contains_downsampled_arrays(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(video_downsample_rate=2)
        dataset._cache = {}
        dataset._cache_size = 1
        dataset._get_s3 = lambda: None
        values = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)

        def build_episode(*_args):
            return values, values

        dataset._build_state_and_actions = build_episode
        states, actions = dataset._get_episode("bucket", "episode/", "bucket/episode/")

        np.testing.assert_array_equal(states, values[::2])
        np.testing.assert_array_equal(actions, states)

    def test_extracts_songling_qpos14_from_canonical55(self):
        canonical = torch.arange(2 * 55, dtype=torch.float32).reshape(2, 55)
        mask = torch.zeros_like(canonical)
        mask[:, SONGLING_QPOS14_INDICES] = 1
        state = ArioStreamingDataset._extract_songling_qpos14(
            {
                "__canonical55__": canonical,
                "__canonical55_mask__": mask,
            }
        )

        expected = arm_joints_deg_to_rad(canonical[:, SONGLING_QPOS14_INDICES].numpy())
        np.testing.assert_allclose(state, expected)
        self.assertEqual(state.shape, (2, 14))
        self.assertNotIn(6, SONGLING_QPOS14_INDICES)
        self.assertNotIn(23, SONGLING_QPOS14_INDICES)

    def test_extracts_songling_qpos14_converts_arm_joints_not_grippers(self):
        canonical = torch.zeros((1, 55), dtype=torch.float32)
        canonical[0, 0] = 180.0
        canonical[0, 16] = 80.0
        canonical[0, 17] = 90.0
        canonical[0, 33] = 12.5
        mask = torch.zeros_like(canonical)
        mask[:, SONGLING_QPOS14_INDICES] = 1

        state = ArioStreamingDataset._extract_songling_qpos14(
            {
                "__canonical55__": canonical,
                "__canonical55_mask__": mask,
            }
        )

        np.testing.assert_allclose(state[0, 0], np.pi)
        np.testing.assert_allclose(state[0, 6], 80.0)
        np.testing.assert_allclose(state[0, 7], np.pi / 2)
        np.testing.assert_allclose(state[0, 13], 12.5)

    def test_rejects_valid_seventh_joint_slot(self):
        canonical = torch.zeros((2, 55), dtype=torch.float32)
        mask = torch.zeros_like(canonical)
        mask[:, SONGLING_QPOS14_INDICES] = 1
        mask[:, 6] = 1

        with self.assertRaisesRegex(ValueError, "seventh joint"):
            ArioStreamingDataset._extract_songling_qpos14(
                {
                    "__canonical55__": canonical,
                    "__canonical55_mask__": mask,
                }
            )

    def test_songling_action_chunk_starts_at_next_frame_and_clamps(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(data_format="songling_canonical55", action_start_offset=1)
        dataset._action_horizon = 3
        action_source = np.arange(4 * 14, dtype=np.float32).reshape(4, 14)

        np.testing.assert_array_equal(
            dataset._get_action_chunk(action_source, frame_idx=1),
            action_source[[2, 3, 3]],
        )

    def test_instruction_segments_use_inclusive_start_and_end_frames(self):
        segments = ArioStreamingDataset._parse_instruction_segments(
            {
                "fine": [
                    {"instruction": "first", "start_frame": 0, "end_frame": 4},
                    {"instruction": "second", "start_frame": 5, "end_frame": 9},
                ]
            },
            "fine",
        )
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(video_downsample_rate=1, task="")
        dataset._instructions = {"episode/": segments}

        self.assertEqual(dataset._get_prompt("episode/", 0), "first")
        self.assertEqual(dataset._get_prompt("episode/", 4), "first")
        self.assertEqual(dataset._get_prompt("episode/", 5), "second")
        self.assertEqual(dataset._get_prompt("episode/", 9), "second")
        self.assertEqual(dataset._get_prompt("episode/", 10), "")

    def test_instruction_lookup_converts_downsampled_index_to_raw_frame(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        dataset._config = ArioConfig(video_downsample_rate=3, task="")
        dataset._instructions = {"episode/": [(6, 8, "segment")]}

        self.assertEqual(dataset._get_prompt("episode/", 2), "segment")
        self.assertEqual(dataset._get_prompt("episode/", 3), "")


if __name__ == "__main__":
    unittest.main()
