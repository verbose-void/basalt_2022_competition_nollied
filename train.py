import logging
import os

import numpy as np
import gym
import minerl

import coloredlogs

from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import VectorizedDynamicsModelEnvironment
from fgz.architecture.dynamics_function import DynamicsFunction, MineRLDynamicsEnvironment
from fgz.training.fgz_trainer import FGZTrainer
from fgz.data_utils.data_handler import DataHandler
from vpt.run_agent import load_agent
from constants import *

coloredlogs.install(logging.DEBUG)


def get_agent():
    print("Loading model", PRETRAINED_AGENT_MODEL_FILE)
    print("with weights", PRETRAINED_AGENT_WEIGHTS_FILE)
    return load_agent(PRETRAINED_AGENT_MODEL_FILE, PRETRAINED_AGENT_WEIGHTS_FILE)


def get_dynamics_function():
    # TODO: should we initialize the weights of the dynamics function with pretrained agent weights of some kind?
    return DynamicsFunction(
        discriminator_classes=NUM_DISCRIMINATOR_CLASSES,
        embedder_layers=16,
        button_features=128,
        camera_features=32,
    )


def get_dynamics_environment(minerl_env: gym.Env) -> MineRLDynamicsEnvironment:
    dynamics_function = get_dynamics_function()
    return MineRLDynamicsEnvironment(
        minerl_env.action_space, 
        dynamics_function=dynamics_function,
        n=NUM_WALKERS,
    )


def get_data_handler(agent):
    return DataHandler(DATASET_PATHS, agent=agent, frames_per_window=UNROLL_STEPS)


def main():
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    All trained models should be placed under "train" directory!
    """

    minerl_env = gym.make('MineRLBasaltFindCave-v0')
    agent = get_agent()
    dynamics_env = get_dynamics_environment(minerl_env)
    fmc = FMC(dynamics_env)
    trainer = FGZTrainer(minerl_env, agent, fmc, unroll_steps=UNROLL_STEPS)

    # For an example, lets just run 100 steps of the environment for training
    # obs = env.reset()
    # for _ in range(100):
    #     obs, reward, done, info = env.step(env.action_space.sample())
    #     # Do your training here
    #     if done:
    #         break

    # # Save trained model to train/ directory
    # # For a demonstration, we save some dummy data.
    # # NOTE: All trained models should be placed under train directory!
    # np.save("./train/parameters.npy", np.random.random((10,)))

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues.
    minerl_env.close()


if __name__ == "__main__":
    main()
