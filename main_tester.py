import os
from fractal_zero.data.tree_sampler import TreeSampler
from fractal_zero.search.fmc import FMC

import minerl
import gym

import cv2
import torch
import numpy as np
from vpt.agent import AGENT_RESOLUTION

from xirl_zero.main_trainer import Trainer, now_filename
from xirl_zero.architecture.dynamics_function import MineRLDynamicsEnvironment
from tqdm import tqdm

class Tester:
    env: gym.Env = None

    def __init__(self, path_to_experiment: str, iteration: int=None, device=None):
        self.path_to_experiment = path_to_experiment
        
        checkpoint_dir = os.path.join(path_to_experiment, "checkpoints")

        # default pick the last iteration
        if iteration is None:
            iterations = [int(fn.split(".")[0]) for fn in os.listdir(checkpoint_dir)]
            iteration = max(iterations)
        
        fn = f"{iteration}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, fn)
        target_state_path = os.path.join(path_to_experiment, "target_states", fn)

        print(f"Loading {fn} checkpoint and target state from {path_to_experiment}")

        trainer: Trainer = torch.load(checkpoint_path, map_location=device)
        target_state: torch.Tensor = torch.load(target_state_path, map_location=device)

        # TODO: handle this better?
        self.device = device if device is not None else trainer.representation_trainer.config.device

        self.minerl_env_id = trainer.config.minerl_env_id

        self.representation_function = trainer.representation_trainer.model
        self.dynamics_function = trainer.dynamics_trainer.model
        self.target_state = target_state

    def load_environment(self, env: gym.Env=None):
        if env is None:
            env = gym.make(self.minerl_env_id)

        actual_env_id = env.unwrapped.spec.id
        if actual_env_id != self.minerl_env_id:
            raise ValueError(f"Cross-task testing is not recommended. The actual env ID loaded was {actual_env_id}, but we expected {self.minerl_env_id}.")

        num_walkers = 16

        self.env = env
        self.dynamics_env = MineRLDynamicsEnvironment(
            self.env.action_space,
            self.dynamics_function, 
            self.target_state,
            n=num_walkers,
        )

        self.fmc = FMC(self.dynamics_env)

    def get_action(self, obs, force_no_escape: bool):
        self.representation_function.eval()
        self.dynamics_function.eval()

        obs = torch.tensor(obs, device=self.device)
        state = self.representation_function.embed(obs).squeeze()
        self.dynamics_env.set_all_states(state)

        # action = self.dynamics_env.action_space.sample()  # TODO FMC
        self.fmc.simulate(16)
        action = self.fmc.tree.best_path.first_action

        if force_no_escape:
            action["ESC"] = 0

        return action

    def play_episode(
        self, 
        min_steps: int, 
        max_steps: int, 
        render: bool = False, 
        smoke_test: bool = False, 
        use_tqdm: bool = True,
        save_video: bool = False,
    ):
        if self.env is None:
            raise ValueError("load_environment must be called first.")

        if smoke_test:
            obs = np.random.uniform(size=(*AGENT_RESOLUTION, 3))
        else:
            obs = self.env.reset()

        if save_video:
            video_dir = os.path.join(self.path_to_experiment, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{now_filename()}.mp4")

            resolution = AGENT_RESOLUTION if smoke_test else (640, 360)
            video_recorder = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, resolution)
            print(f"Saving video at {video_path}")

        for step in tqdm(range(max_steps), desc=f"Playing {self.minerl_env_id} Episode", disable=not use_tqdm):
            action = self.get_action(obs, force_no_escape=step < min_steps)

            if smoke_test:
                obs = np.random.uniform(size=(*AGENT_RESOLUTION, 3))
                reward = 0
                done = False
                info = {}
            else:
                obs, reward, done, info = self.env.step(action)

            if save_video:
                if smoke_test:
                    frame = obs[..., ::-1].astype(np.uint8)
                else:
                    frame = obs["pov"][..., ::-1]

                video_recorder.write(frame)

            if render:
                self.env.render()

            if done:
                break

        self.env.close()

if __name__ == "__main__":
    tester = Tester("./train/xirl_zero/MineRLBasaltMakeWaterfall-v0/2022-10-28_02-52-40_PM")
    tester.load_environment()
    tester.play_episode(min_steps=16, max_steps=64, render=False, smoke_test=True, save_video=True)
