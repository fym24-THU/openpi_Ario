import pathlib

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config


def test_songling_data_config_uses_canonical55_and_next_frame_actions(tmp_path: pathlib.Path):
    model_config = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50)
    data_config = config.ArioSonglingDataConfig(
        repo_id="songling/test",
        s3_prefixes="s3://bucket/songling/",
    ).create(tmp_path, model_config)

    assert data_config.ario_config.data_format == "songling_canonical55"
    assert data_config.ario_config.action_start_offset == 1
    assert data_config.ario_config.min_frames == 51
    assert data_config.ario_config.instruction_field == "sub_instructions"
    assert data_config.ario_config.filter_episodes_by_state
    assert isinstance(data_config.data_transforms.inputs[0], config.songling_policy.SonglingInputs)
    assert isinstance(data_config.data_transforms.inputs[1], transforms.DeltaActions)
    assert data_config.data_transforms.inputs[1].mask == transforms.make_bool_mask(6, -1, 6, -1)
    assert isinstance(data_config.model_transforms.inputs[-1], transforms.PadStatesAndActions)
    assert data_config.model_transforms.inputs[-1].model_action_dim == 32


def test_songling_debug_train_config_is_registered():
    train_config = config.get_config("pi05_songling_fold_ario_debug")

    assert isinstance(train_config.data, config.ArioSonglingDataConfig)
    assert train_config.data.instruction_field == "sub_instructions"
    assert train_config.data.default_prompt == ""
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 50
    assert train_config.max_checkpoints_to_keep is None


def test_songling_garment_folding_config_contains_all_task_prefixes():
    train_config = config.get_config("pi05_songling_garment_folding_ario")

    assert isinstance(train_config.data, config.ArioSonglingDataConfig)
    prefixes = train_config.data.s3_prefixes.split(",")
    assert len(prefixes) == len(config.SONGLING_GARMENT_FOLDING_TASKS) == 14
    assert prefixes == [
        f"{config.SONGLING_GARMENT_FOLDING_ROOT}/{task}/" for task in config.SONGLING_GARMENT_FOLDING_TASKS
    ]
    assert train_config.data.repo_id == "songling/garment_folding"
    assert train_config.data.max_episodes is None
    assert train_config.data.filter_episodes_by_state
    assert train_config.data.disk_cache_max_gb == 1_000.0
    assert train_config.data.instruction_field == "sub_instructions"
    assert train_config.num_train_steps == 100_000
    assert train_config.max_checkpoints_to_keep is None


def test_songling_garment_folding_config_excludes_corrupt_episodes(tmp_path: pathlib.Path):
    train_config = config.get_config("pi05_songling_garment_folding_ario")
    assert isinstance(train_config.data, config.ArioSonglingDataConfig)
    assert train_config.data.excluded_episodes == config.SONGLING_GARMENT_FOLDING_EXCLUDED_EPISODES
    assert train_config.data.episode_frames_per_batch == 16

    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.ario_config is not None
    assert data_config.ario_config.excluded_episodes == config.SONGLING_GARMENT_FOLDING_EXCLUDED_EPISODES
    assert data_config.ario_config.episode_frames_per_batch == 16
