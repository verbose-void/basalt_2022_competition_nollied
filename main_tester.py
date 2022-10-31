from argparse import ArgumentParser
import os
from fractal_zero.data.tree_sampler import TreeSampler
from fractal_zero.search.fmc import FMC

import minerl
import gym

import cv2
import torch
import numpy as np
from vpt.agent import AGENT_RESOLUTION
from fgz.architecture.xirl_model import XIRLModel

from xirl_zero.main_trainer import Trainer, now_filename
from xirl_zero.architecture.dynamics_function import DynamicsFunction, MineRLDynamicsEnvironment
from tqdm import tqdm

from xirl_zero.search.dynamics_fmc import DynamicsFMC

class Tester:
    env: gym.Env = None
    representation_function: XIRLModel
    dynamics_function: DynamicsFunction

    def __init__(self, path_to_experiment: str, iteration: int=None, device=None):
        self.path_to_experiment = path_to_experiment
        
        checkpoint_dir = os.path.join(path_to_experiment, "checkpoints")

        # default pick the last iteration
        if iteration is None:
            iterations = [int(fn.split(".")[0]) for fn in os.listdir(checkpoint_dir)]
            iteration = max(iterations)
        self.iteration = iteration
        
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

        self.new_video_paths = []

    def load_environment(self, env: gym.Env=None):
        if env is None:
            env = gym.make(self.minerl_env_id)

        actual_env_id = env.unwrapped.spec.id
        if actual_env_id != self.minerl_env_id:
            raise ValueError(f"Cross-task testing is not recommended. The actual env ID loaded was {actual_env_id}, but we expected {self.minerl_env_id}.")

        self.env = env

    def get_action(self, obs, force_no_escape: bool):
        self.representation_function.eval()
        self.dynamics_function.eval()

        # x = torch.tensor(obs["pov"], device=self.device)
        x = self.representation_function.prepare_observation(obs).to(self.device)
        state = self.representation_function.embed(x).squeeze()

        fmc = DynamicsFMC(
            self.dynamics_function, 
            self.target_state, 
            self.env.action_space,
            num_walkers=512,
            steps=128,
            balance=3.0,
        )

        actions = fmc.get_actions(state)

        action_percent = 0.5
        max_action_index = max(1, int(np.ceil(len(actions) * action_percent)))
        print("total actions", len(actions), "max", max_action_index)

        actions = actions[:max_action_index].flatten().tolist()

        for action in actions:
            if force_no_escape:
                action["ESC"] = 0

        return actions

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
            video_path = os.path.join(video_dir, f"{self.iteration}___{now_filename()}.mp4")

            resolution = AGENT_RESOLUTION if smoke_test else (640, 360)
            video_recorder = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, resolution)
            print(f"Saving video at {video_path}")
            
            self.new_video_paths.append(video_path)

        self.play_step = 0

        def _after_step():
            if save_video:
                if smoke_test:
                    frame = obs[..., ::-1].astype(np.uint8)
                else:
                    frame = obs["pov"][..., ::-1]

                video_recorder.write(frame)

            if render:
                self.env.render()

            self.play_step += 1

        # for step in tqdm(range(max_steps), desc=f"Playing {self.minerl_env_id} Episode", disable=not use_tqdm):
        while True:
            actions = self.get_action(obs, force_no_escape=self.play_step < min_steps)

            if smoke_test:
                obs = np.random.uniform(size=(*AGENT_RESOLUTION, 3))
                reward = 0
                done = False
                info = {}

                _after_step()
            else:
                for action in actions:
                    obs, reward, done, info = self.env.step(action)

                    _after_step()

                    if done:
                        break
                
            if done or self.play_step >= max_steps:
                break

        self.env.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--experiment-directory", type=str)
    parser.add_argument("--iteration", type=int, default=None, help="The checkpoint iteration to use.")

    parser.add_argument("--min-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=64)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")

    args = dict(parser.parse_args().__dict__)
    
    # tester = Tester("./train/xirl_zero/MineRLBasaltMakeWaterfall-v0/2022-10-28_02-52-40_PM")

    experiment_dir = args.pop("experiment_directory")

    device = None
    if args.pop("force_cpu"):
        device = torch.device("cpu")

    tester = Tester(experiment_dir, iteration=args.pop("iteration"), device=device)

    tester.load_environment()
    tester.play_episode(**args)
