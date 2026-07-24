import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_episode_aware_distributed_sampler_preserves_locality():
    episode_lengths = (6, 4, 5)
    rank0 = _data_loader.EpisodeAwareDistributedSampler(
        episode_lengths, num_replicas=2, rank=0, shuffle=False, drop_last=True
    )
    rank1 = _data_loader.EpisodeAwareDistributedSampler(
        episode_lengths, num_replicas=2, rank=1, shuffle=False, drop_last=True
    )

    assert list(rank0) == [0, 2, 4, 6, 8, 10, 12]
    assert list(rank1) == [1, 3, 5, 7, 9, 11, 13]


def test_episode_aware_distributed_sampler_shuffles_by_epoch():
    sampler = _data_loader.EpisodeAwareDistributedSampler(
        (8, 8, 8), num_replicas=2, rank=0, shuffle=True, seed=7, drop_last=True
    )

    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert epoch0 != epoch1
    assert len(epoch0) == len(epoch1) == 12

    def episode_id(index: int) -> int:
        return index // 8

    # Once sampling leaves an episode, it should not return to it in the same epoch.
    episode_runs = [episode_id(epoch0[0])]
    for index in epoch0[1:]:
        current = episode_id(index)
        if current != episode_runs[-1]:
            episode_runs.append(current)
    assert len(episode_runs) == len(set(episode_runs))


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
