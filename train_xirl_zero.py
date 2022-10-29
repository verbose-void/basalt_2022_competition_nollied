

from warnings import warn
import wandb
from xirl_zero.main_trainer import Config, Trainer

import torch
from xirl_zero.trainers.muzero_dynamics import MuZeroDynamicsConfig

from xirl_zero.trainers.tcc_representation import TCCConfig


# TODO: determine based on task idx
# DATASET_PATH = "/Volumes/CORSAIR/data/MineRLBasaltMakeWaterfall-v0"
# DATASET_PATH = "./data/MineRLBasaltMakeWaterfall-v0"
MINERL_ENV_ID = "MineRLBasaltMakeWaterfall-v0"

OUTPUT_DIR = f"./train/xirl_zero/{MINERL_ENV_ID}"

SMOKE_TEST = False
USE_WANDB = True

# NOTE: a smoke test is basically a very fast run through the entire train script with the aim of catching runtime errors.
SMOKE_TEST_CONFIG = Config(
    minerl_env_id=MINERL_ENV_ID,
    train_steps=2,
    eval_every=1,
    eval_steps=2,
    checkpoint_every=1,
    max_frames=8,
    max_trajectories=8,
    use_wandb=USE_WANDB,
    model_log_frequency=1,
    num_frame_samples=8,
    verbose=True,

    representation_config=TCCConfig(
        num_unfrozen_layers=4,  # number of embedder layers to have gradients
    ),
)

CONFIG = Config(
    minerl_env_id=MINERL_ENV_ID,
    train_steps=10_000,
    eval_every=100,
    eval_steps=0,
    checkpoint_every=100,
    max_frames=None,
    max_trajectories=None,
    use_wandb=USE_WANDB,
    model_log_frequency=10,

    num_frame_samples=128,
    representation_config=TCCConfig(
        learning_rate=0.000181,
        batch_size=32,
        embed_batch_size=128,
        num_unfrozen_layers=4,  # number of embedder layers to have gradients
    ),
    dynamics_config=MuZeroDynamicsConfig(),
)

def run_train_loop(config: Config):
    trainer = Trainer(config)

    def run_eval(config: Config):
        for _ in range(config.eval_steps):
            trainer.eval_step()

    def run_train(config: Config):
        for step in range(config.train_steps):
            trainer.train_step()

            is_last_step = step == (config.train_steps - 1)

            if (step + 1) % config.eval_every == 0 or is_last_step:
                run_eval(config)

            if (step + 1) % config.checkpoint_every == 0 or is_last_step:
                trainer.checkpoint(OUTPUT_DIR)
                _, target_state = trainer.generate_and_save_target_state(OUTPUT_DIR)

    print("\n\n\n\nTraining with config:", config.asdict(), "\n\n\n\n")
    run_train(config)

if __name__ == "__main__":
    config = SMOKE_TEST_CONFIG if SMOKE_TEST else CONFIG

    if SMOKE_TEST:
        warn("\n\n\n\n\n\nWARNING: DOING A SMOKE TEST!\n\n\n\n\n\n")

    if config.use_wandb:
        project = f"xirl_zero_{MINERL_ENV_ID}"
        if SMOKE_TEST:
            project += "_smoke"
        wandb.init(project=project, config=config.asdict())

    run_train_loop(config)
