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

        num_walkers = 256

        self.env = env
        self.dynamics_env = MineRLDynamicsEnvironment(
            self.env.action_space,
            self.dynamics_function, 
            self.target_state,
            n=num_walkers,
        )

        self.fmc = FMC(self.dynamics_env, reward_is_score=True, balance=3)

    def get_action(self, obs, force_no_escape: bool):
        self.representation_function.eval()
        self.dynamics_function.eval()

        self.fmc.reset()

        # x = torch.tensor(obs["pov"], device=self.device)
        x = self.representation_function.prepare_observation(obs).to(self.device)
        state = self.representation_function.embed(x).squeeze()
        self.dynamics_env.set_all_states(state)

        self.fmc.simulate(128, use_tqdm=True)

        best_path = self.fmc.tree.best_path
        # action = best_path.first_action

        actions = []
        for state, action in best_path:
            if force_no_escape:
                action["ESC"] = 0
            actions.append(action)

        distance_to_target = best_path.ordered_states[1].reward
        average_distance_to_target = best_path.average_reward

        # TODO: if current state embedding is within a certain threshold of the target state, force ESC action.
        # print(action)
        print(len(best_path.ordered_states))
        print(len(actions), "actions")
        print(distance_to_target, average_distance_to_target)
        print("\n\n")

        assert len(actions) > 0

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
            video_path = os.path.join(video_dir, f"{now_filename()}.mp4")

            resolution = AGENT_RESOLUTION if smoke_test else (640, 360)
            video_recorder = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, resolution)
            print(f"Saving video at {video_path}")
            
            self.new_video_paths.append(video_path)

        def _after_step():
            if save_video:
                if smoke_test:
                    frame = obs[..., ::-1].astype(np.uint8)
                else:
                    frame = obs["pov"][..., ::-1]

                video_recorder.write(frame)

            if render:
                self.env.render()

        for step in tqdm(range(max_steps), desc=f"Playing {self.minerl_env_id} Episode", disable=not use_tqdm):
            actions = self.get_action(obs, force_no_escape=step < min_steps)

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
                
            if done:
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
