import dataclasses
from unittest import mock

import jax
import numpy as np

from openpi.datasets.ario_dataset import ArioConfig
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _EpisodeIndexDataset:
    episode_lengths = (5, 7)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, index):
        return {"index": np.int64(index)}


def test_episode_aware_sampler_groups_contiguous_frames_with_wraparound():
    sampler = _data_loader.EpisodeAwareBatchSampler(
        [5, 7],
        batch_size=4,
        frames_per_episode=2,
        seed=7,
    )

    for batch in sampler:
        assert len(batch) == 4
        for group in (batch[:2], batch[2:]):
            if group[0] < 5:
                local = group
                episode_length = 5
            else:
                local = [index - 5 for index in group]
                episode_length = 7
            assert local[1] == (local[0] + 1) % episode_length


def test_episode_aware_sampler_weights_episodes_by_frame_count():
    sampler = _data_loader.EpisodeAwareBatchSampler(
        [10, 30],
        batch_size=4,
        frames_per_episode=1,
        seed=11,
    )

    long_episode_samples = 0
    total_samples = 0
    for _ in range(500):
        for batch in sampler:
            long_episode_samples += sum(index >= 10 for index in batch)
            total_samples += len(batch)

    long_episode_fraction = long_episode_samples / total_samples
    assert 0.72 < long_episode_fraction < 0.78


def test_episode_aware_sampler_is_reproducible_across_epochs():
    first = _data_loader.EpisodeAwareBatchSampler(
        [20, 30],
        batch_size=10,
        frames_per_episode=5,
        seed=42,
    )
    second = _data_loader.EpisodeAwareBatchSampler(
        [20, 30],
        batch_size=10,
        frames_per_episode=5,
        seed=42,
    )

    first_epoch = list(first)
    assert first_epoch == list(second)
    second_epoch = list(first)
    assert second_epoch == list(second)
    assert second_epoch != first_epoch


def test_torch_data_loader_accepts_episode_batch_sampler():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)
    sampler = _data_loader.EpisodeAwareBatchSampler(
        [8, 8],
        batch_size=4,
        frames_per_episode=2,
        seed=5,
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        batch_sampler=sampler,
        num_batches=2,
        num_workers=2,
    )

    batches = list(loader)
    assert len(batches) == 2
    assert all(x.shape[0] == 4 for batch in batches for x in jax.tree.leaves(batch))


def test_create_torch_data_loader_enables_episode_aware_sampler():
    data_config = _config.DataConfig(
        repo_id="test",
        ario_config=ArioConfig(episode_frames_per_batch=2),
    )
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)

    with mock.patch.object(_data_loader, "create_torch_dataset", return_value=_EpisodeIndexDataset()):
        loader = _data_loader.create_torch_data_loader(
            data_config,
            model_config,
            action_horizon=50,
            batch_size=4,
            skip_norm_stats=True,
            shuffle=True,
            num_batches=2,
        )
        batches = list(loader._data_loader)  # noqa: SLF001

    assert len(batches) == 2
    for batch in batches:
        indices = np.asarray(batch["index"])
        for group in (indices[:2], indices[2:]):
            if group[0] < 5:
                local = group
                episode_length = 5
            else:
                local = group - 5
                episode_length = 7
            assert local[1] == (local[0] + 1) % episode_length


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
