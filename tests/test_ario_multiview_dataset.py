"""Tests for Ario multi-view discovery and cache isolation."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from openpi.datasets.ario_dataset import ArioConfig, ArioStreamingDataset


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

    def test_cache_identity_separates_old_and_skip_video_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
            dataset._config = ArioConfig(
                disk_cache_dir=tmpdir,
                multi_view=True,
                skip_video=False,
            )
            cache_key = "bucket/root/episode/"
            training_path = dataset._disk_cache_path(cache_key)

            dataset._config.skip_video = True
            norm_stats_path = dataset._disk_cache_path(cache_key)

            legacy_path = Path(tmpdir) / f"{hashlib.md5(cache_key.encode()).hexdigest()}.npz"
            self.assertNotEqual(training_path, norm_stats_path)
            self.assertNotEqual(training_path, legacy_path)

    def test_missing_head_view_never_falls_back_to_composed_video(self):
        dataset = ArioStreamingDataset.__new__(ArioStreamingDataset)
        requested_keys = []

        def fail_download(_s3, _bucket, video_key):
            requested_keys.append(video_key)
            raise FileNotFoundError(video_key)

        dataset._extract_video_frames = fail_download

        with self.assertRaisesRegex(RuntimeError, "cam_high"):
            dataset._extract_multi_view_frames(None, "bucket", "root/episode/")
        self.assertEqual(
            requested_keys,
            ["root/episode/raw_video/cam_high.mp4"],
        )


if __name__ == "__main__":
    unittest.main()
