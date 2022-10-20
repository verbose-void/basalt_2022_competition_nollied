from argparse import ArgumentParser
import logging
import os

import numpy as np
import gym
import minerl
import torch
from tqdm import tqdm

import coloredlogs

from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import VectorizedDynamicsModelEnvironment
from fgz.architecture.dynamics_function import (
    DynamicsFunction,
    MineRLDynamicsEnvironment,
)
from fgz.training.fgz_trainer import FGZTrainer
from fgz.data_utils.data_handler import DataHandler
from vpt.run_agent import load_agent
from fgz_config import FGZConfig

try:
    import wandb
except ImportError:
    pass  # optional

coloredlogs.install(logging.DEBUG)


def get_agent(config: FGZConfig):
    print("Loading model", config.model_filename)
    print("with weights", config.weights_filename)
    return load_agent(config.model_path, config.weights_path)


def get_dynamics_function(config: FGZConfig):
    # TODO: should we initialize the weights of the dynamics function with pretrained agent weights of some kind?
    return DynamicsFunction(
        discriminator_classes=config.num_discriminator_classes,
        embedder_layers=4,
        button_features=128,
        camera_features=128,
    )


def get_dynamics_environment(config: FGZConfig) -> MineRLDynamicsEnvironment:
    dynamics_function = get_dynamics_function(config)
    return MineRLDynamicsEnvironment(
        config.action_space,
        dynamics_function=dynamics_function,
        n=config.num_walkers,
    )


def get_data_handler(config: FGZConfig, agent):
    return DataHandler(
        config.dataset_paths, agent=agent, frames_per_window=config.unroll_steps
    )


def run_training(trainer, lr_scheduler, train_steps: int, batch_size: int, checkpoint_every: int = 10):

    for train_step in tqdm(range(train_steps), desc="Training"):
        trainer.train_sub_trajectories(batch_size=batch_size, use_tqdm=False)

        if train_step % checkpoint_every == 0:
            trainer.save("./train/fgz_dynamics_checkpoint.pth")

        if lr_scheduler is not None:
            lr_scheduler.step()

def main(use_wandb: bool):
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    All trained models should be placed under "train" directory!
    """

    train_steps = 100
    batch_size = 8

    enabled_tasks = [2, 3]  # cave and waterfall 
    # enabled_tasks = [0, 1, 2, 3]  # all 

    config = FGZConfig(
        enabled_tasks=enabled_tasks,
        disable_fmc_detection=True,  # if true, only classification will occur. 
        use_wandb=use_wandb,
        unroll_steps=64,
    )

    # minerl_env = gym.make('MineRLBasaltMakeWaterfall-v0')
    agent = get_agent(config)
    dynamics_env = get_dynamics_environment(config)
    data_handler = get_data_handler(config, agent)

    # setup optimizer and learning rate schedule
    dynamics_function_optimizer = torch.optim.Adam(
        dynamics_env.dynamics_function.parameters(),
        lr=0.001,
        # weight_decay=1e-4,
    )
    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(dynamics_function_optimizer, step_size=10, gamma=0.95)

    # setup training/fmc objects
    fmc = FMC(dynamics_env, freeze_best=True)
    trainer = FGZTrainer(agent, fmc, data_handler, dynamics_function_optimizer, config=config)

    if config.use_wandb:
        wandb.init(project="fgz-v0.1.1", config=config.asdict())

    run_training(trainer, lr_scheduler, train_steps=train_steps, batch_size=batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--use-wandb", action="store_true", help="Enables usage of weights and biases.")

    args = parser.parse_args().__dict__
    main(**args)
