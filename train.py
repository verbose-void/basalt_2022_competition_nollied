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

coloredlogs.install(logging.DEBUG)

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
DATASET_PATHS = [
    os.path.join(MINERL_DATA_ROOT, "MineRLBasaltBuildVillageHouse-v0"),
    os.path.join(MINERL_DATA_ROOT, "MineRLBasaltCreateVillageAnimalPen-v0"),
    os.path.join(MINERL_DATA_ROOT, "MineRLBasaltFindCave-v0"),
    os.path.join(MINERL_DATA_ROOT, "MineRLBasaltMakeWaterfall-v0"),
]

VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")
PRETRAINED_AGENT_MODEL_FILE = os.path.join(VPT_MODELS_ROOT, "foundation-model-2x.model")
PRETRAINED_AGENT_WEIGHTS_FILE = os.path.join(VPT_MODELS_ROOT, "rl-from-early-game-2x.weights")


NUM_WALKERS = 8
DISCRIMINATOR_CLASSES = 2  # TODO: set discriminator classes to num of tasks
TARGET_LOGIT = 0
UNROLL_STEPS = 8


def get_agent():
    return load_agent(PRETRAINED_AGENT_MODEL_FILE, PRETRAINED_AGENT_WEIGHTS_FILE)


def get_dynamics_function():
    # TODO: should we initialize the weights of the dynamics function with pretrained agent weights of some kind?
    return DynamicsFunction(
        discriminator_classes=DISCRIMINATOR_CLASSES,
    )


def get_dynamics_environment(minerl_env: gym.Env) -> MineRLDynamicsEnvironment:
    dynamics_function = get_dynamics_function()
    return MineRLDynamicsEnvironment(
        minerl_env.action_space, 
        dynamics_function=dynamics_function, 
        target_discriminator_logit=TARGET_LOGIT,
        n=NUM_WALKERS,
    )


def get_data_handler():
    return DataHandler(DATASET_PATHS)


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
