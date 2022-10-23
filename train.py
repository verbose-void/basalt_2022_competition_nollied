from argparse import ArgumentParser
import logging
import os
from typing import List

import numpy as np
import gym
import minerl
import torch
from tqdm import tqdm

import coloredlogs

from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import VectorizedDynamicsModelEnvironment
from vpt.agent import MineRLAgent
from fgz.architecture.dynamics_function import (
    DynamicsFunction,
    MineRLDynamicsEnvironment,
)
from fgz.loading import get_agent
from fgz.training.fgz_trainer import FGZTrainer
from fgz.data_utils.data_handler import DataHandler
from fgz_config import TASKS, FGZConfig

try:
    import wandb
except ImportError:
    pass  # optional

coloredlogs.install(logging.DEBUG)


def get_dynamics_function(config: FGZConfig):
    # TODO: should we initialize the weights of the dynamics function with pretrained agent weights of some kind?
    return DynamicsFunction(
        state_embedding_size=2048,  # TODO: make automatic
        discriminator_classes=config.num_discriminator_classes,
        embedder_layers=4,
        button_features=128,
        camera_features=128,
    )


def get_dynamics_environment(config: FGZConfig, agent: MineRLAgent) -> MineRLDynamicsEnvironment:
    dynamics_function = get_dynamics_function(config)

    return MineRLDynamicsEnvironment(
        config.action_space, dynamics_function=dynamics_function, agent=agent, n=config.num_walkers
    )


def get_data_handler(config: FGZConfig, agent):
    return DataHandler(
        config.dataset_paths, agent=agent, frames_per_window=config.unroll_steps
    )


def run_training(
    trainer, lr_scheduler, train_steps: int, batch_size: int, checkpoint_every: int = 10, evaluate_save_video_every: int = 100
):

    best_score = 0.0

    for train_step in tqdm(range(train_steps), desc="Training"):
        score = trainer.train_sub_trajectories(batch_size=batch_size, use_tqdm=False)

        if train_step % checkpoint_every == 0:
            trainer.save("./train/checkpoints")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if score >= best_score:
            best_score = score
            trainer.save("./train/checkpoints/", f"./train/checkpoints/{trainer.run_name}_best.pth")

        if (train_step + 1) % evaluate_save_video_every== 0:
            task_id = trainer.config.enabled_tasks[0]
            eval_env_id = TASKS[task_id]["dataset_dir"]
            trainer.evaluate(eval_env_id, render=False, save_video=True, max_steps=32, force_no_escape=True)

def main(
    use_wandb: bool, 
    fmc_logit: bool, 
    batch_size: int, 
    unroll_steps: int,
    train_steps: int, 
    tasks: List[int],
    fmc_steps: int,
    num_walkers: int,
    fmc_random_policy: bool,
    learning_rate: float,
    consistency_loss_coeff: float,
):
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    All trained models should be placed under "train" directory!
    """

    # enabled_tasks = [2]  # cave only
    # enabled_tasks = [2, 3]  # cave and waterfall
    # enabled_tasks = [0, 1, 2, 3]  # all

    config = FGZConfig(
        model_filename="foundation-model-2x.model",
        weights_filename="rl-from-early-game-2x.weights",
        enabled_tasks=tasks,
        disable_fmc_detection=not fmc_logit,  # if true, only classification will occur.
        use_wandb=use_wandb,
        verbose=True,
        unroll_steps=unroll_steps,
        fmc_steps=fmc_steps,
        num_walkers=num_walkers,
        fmc_random_policy=fmc_random_policy,
        learning_rate=learning_rate,
        batch_size=batch_size,
        consistency_loss_coeff=consistency_loss_coeff,
    )

    print(f"Running with config: {config}")
    if config.use_wandb:
        wandb.init(project="fgz_all_tasks", config=config.asdict())

    # minerl_env = gym.make('MineRLBasaltMakeWaterfall-v0')
    agent = get_agent(config)
    dynamics_env = get_dynamics_environment(config, agent)
    data_handler = get_data_handler(config, agent)

    # setup optimizer and learning rate schedule
    dynamics_function_optimizer = torch.optim.Adam(
        dynamics_env.dynamics_function.parameters(),
        lr=config.learning_rate,
        # weight_decay=1e-4,
    )
    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(dynamics_function_optimizer, step_size=10, gamma=0.95)

    # setup training/fmc objects
    fmc = FMC(dynamics_env, freeze_best=True)
    trainer = FGZTrainer(
        agent, fmc, data_handler, dynamics_function_optimizer, config=config
    )

    run_training(trainer, lr_scheduler, train_steps=train_steps, batch_size=config.batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--use-wandb", action="store_true", help="Enables usage of weights and biases."
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--consistency-loss-coeff", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.00008)
    parser.add_argument("--unroll-steps", type=int, default=4)

    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument('--tasks', nargs="+", type=int, help="List of integers that correspond to the enabled tasks.", default=[2, 3])

    # FMC hyperparameters
    parser.add_argument("--num-walkers", type=int, default=128, help="Number of simultaneous states to be explored in the FMC lookahead search.")
    parser.add_argument("--fmc-logit", action="store_true", help="Improve the task classifier by having it train on FMC data that's exploiting it's neurons like an adversarial setup.")
    parser.add_argument("--fmc-steps", type=int, default=8, help="Number of simulation steps in the FMC lookahead search.")
    parser.add_argument("--fmc-random-policy", action="store_true", help="If true, FMC will not use the agent's policy, instead it will sample random actions.")

    args = parser.parse_args().__dict__

    args["tasks"] = list(args["tasks"])

    main(**args)
