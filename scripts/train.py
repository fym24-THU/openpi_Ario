import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def validate_and_log_multiview_batch(batch: tuple) -> None:
    """Fail fast unless the first transformed batch contains three valid views."""
    observation, actions = batch
    expected_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    missing_images = [key for key in expected_keys if key not in observation.images]
    missing_masks = [key for key in expected_keys if key not in observation.image_masks]
    if missing_images or missing_masks:
        raise RuntimeError(
            "Invalid multi-view batch: "
            f"missing images={missing_images}, missing masks={missing_masks}"
        )

    images = {}
    for key in expected_keys:
        image = np.asarray(observation.images[key])
        mask = np.asarray(observation.image_masks[key]).astype(bool).reshape(-1)
        images[key] = image

        if image.ndim != 4 or tuple(image.shape[1:]) not in {
            (224, 224, 3),
            (3, 224, 224),
        }:
            raise RuntimeError(
                f"Invalid multi-view image shape for {key}: {tuple(image.shape)}; "
                "expected [B,224,224,3] or [B,3,224,224]"
            )
        if mask.size != image.shape[0] or not bool(mask.all()):
            raise RuntimeError(
                f"Invalid multi-view mask for {key}: shape={tuple(mask.shape)}, "
                f"values={mask.tolist()}"
            )

        image_float = image.astype(np.float32)
        per_sample_std = image_float.reshape(image.shape[0], -1).std(axis=1)
        constant_samples = np.where(per_sample_std == 0)[0].tolist()
        if constant_samples:
            raise RuntimeError(
                f"Invalid multi-view image for {key}: constant samples at batch indices "
                f"{constant_samples}"
            )
        image_std = float(image_float.std())
        logging.info(
            "First batch view %s: shape=%s dtype=%s range=[%.3f, %.3f] "
            "std=%.3f mask_all_true=%s",
            key,
            tuple(image.shape),
            image.dtype,
            float(image_float.min()),
            float(image_float.max()),
            image_std,
            bool(mask.all()),
        )

    for left, right in (
        ("base_0_rgb", "left_wrist_0_rgb"),
        ("base_0_rgb", "right_wrist_0_rgb"),
        ("left_wrist_0_rgb", "right_wrist_0_rgb"),
    ):
        equal_samples = (images[left] == images[right]).reshape(images[left].shape[0], -1).all(axis=1)
        duplicate_indices = np.where(equal_samples)[0].tolist()
        if duplicate_indices:
            raise RuntimeError(
                f"Invalid multi-view batch: {left} and {right} are exactly identical "
                f"at batch indices {duplicate_indices}"
            )
        mean_abs_diff = float(np.abs(images[left].astype(np.float32) - images[right].astype(np.float32)).mean())
        logging.info(
            "First batch view difference %s vs %s: mean_abs_diff=%.3f",
            left,
            right,
            mean_abs_diff,
        )

    logging.info(
        "MULTI-VIEW FIRST BATCH CHECK PASSED: batch=%d actions_shape=%s",
        images[expected_keys[0]].shape[0],
        tuple(np.asarray(actions).shape),
    )


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log multi-view status
    if hasattr(config.data, 'multi_view'):
        multi_view_enabled = config.data.multi_view
        if multi_view_enabled:
            logging.info("Multi-view (3-view) training: ENABLED")
            logging.info("  Camera views: cam_high, cam_left_wrist, cam_right_wrist")
            logging.info("  Primary view (observation/image): cam_high (source: raw_video/cam_high.mp4)")
            logging.info("  Only episodes containing all three raw camera videos are used")
        else:
            logging.info("Multi-view (3-view) training: DISABLED (single-view: video.mp4)")

    # Validate first batch for multi-view correctness
    if getattr(config.data, "multi_view", False):
        validate_and_log_multiview_batch(batch)

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Keep a plain-text metrics log beside the checkpoints. The OSS uploader
    # synchronizes this file on every poll.
    loss_log_path = config.checkpoint_dir / "loss.txt"
    if not resuming:
        loss_log_path.write_text("")

    infos = []
    start_time = time.time()
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_start = time.time()
            train_state, info = ptrain_step(train_rng, train_state, batch)
            jax.block_until_ready(info)
            train_time = time.time() - train_start
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())

            # Compute ETA
            elapsed = time.time() - start_time
            time_per_step = elapsed / config.log_interval if config.log_interval > 0 else 0
            remaining_steps = config.num_train_steps - step
            eta_seconds = remaining_steps * time_per_step
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            eta_str = f"ETA={eta_h}h{eta_m:02d}m"

            pbar.write(f"Step {step}: {info_str} train_time={train_time:.3f}s time={elapsed:.1f}s {eta_str}")
            wandb.log(reduced_info, step=step)
            with loss_log_path.open("a") as loss_log:
                loss_log.write(
                    f"Step {step}: "
                    f"grad_norm={reduced_info['grad_norm']:.4f}, "
                    f"loss={reduced_info['loss']:.4f}, "
                    f"param_norm={reduced_info['param_norm']:.4f}\n"
                )
            infos = []
            start_time = time.time()

        data_start = time.time()
        batch = next(data_iter)
        data_time = time.time() - data_start

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
