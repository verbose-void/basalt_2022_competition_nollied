

from dataclasses import dataclass
import os
import numpy as np

from typing import Dict, List

import torch

from new_fgz.architecture.representation_function import RepresentationFunction
from new_fgz.architecture.dynamics_function import DynamicsFunction, vectorize_minerl_actions
from new_fgz.data_utils.contiguous_trajectory_loader import ContiguousTrajectoryLoader


MINERL_DATA_ROOT = os.getenv("MINERL_DATA_ROOT", "data/")
VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")


@dataclass
class FinalConfig:
    minerl_env_ids: List[str]

    # model_filename: str = "foundation-model-1x.model"
    model_filename: str = "foundation-model-2x.model"
    # model_filename: str = "foundation-model-3x.model"

    # weights_filename: str = "foundation-model-1x.weights"
    weights_filename: str = "rl-from-early-game-2x.weights"

    # train_steps: int
    # eval_every: int
    # eval_steps: int
    # checkpoint_every: int

    trajectories_per_batch: int = 4
    num_frame_samples: int = 8

    verbose: bool = True

    use_wandb: bool = False

    # used for smoke tests
    max_frames: int = None
    max_trajectories: int = None
    model_log_frequency: int = 1000

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # representation_config: TCCConfig = field(default_factory=TCCConfig)
    # dynamics_config: MuZeroDynamicsConfig = field(default_factory=MuZeroDynamicsConfig)

    # def asdict(self):
    #     d = dict(self.__dict__)

    #     d.pop("representation_config")
    #     d.pop("dynamics_config")
    #     d["representation_config"] = self.representation_config.__dict__
    #     d["dynamics_config"] = self.dynamics_config.__dict__

    #     return d

    @property
    def num_discriminator_classes(self):
        return len(self.minerl_env_ids) + 1

    @property
    def model_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.model_filename)

    @property
    def weights_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.weights_filename)


class FinalTrainer:

    def __init__(self, config: FinalConfig):
        self.config = config

        self.representation_function = RepresentationFunction(
            self.config.model_path, 
            self.config.weights_path, 
            device=self.config.device,
        )

        self.dynamics_function = DynamicsFunction(2048, discriminator_classes=self.config.num_discriminator_classes)

        self._build_dataloaders()

    def _build_dataloaders(self):
        self.train_loaders: Dict[str, ContiguousTrajectoryLoader] = {}
        self.eval_loaders: Dict[str, ContiguousTrajectoryLoader] = {}
        for env_id in self.config.minerl_env_ids:
            data_path = os.path.join(MINERL_DATA_ROOT, env_id)
            tr, ev = ContiguousTrajectoryLoader.get_train_and_eval_loaders(data_path, max_trajectories=self.config.max_trajectories)
            self.train_loaders[env_id] = tr
            self.eval_loaders[env_id] = ev

    def get_batch(self):
        tasks = []
        all_frames = []
        all_actions = []

        for _ in range(self.config.trajectories_per_batch):
            task = np.random.choice(self.config.minerl_env_ids)
            frames, actions_inbetween = self.train_loaders[task].sample(self.config.num_frame_samples, self.config.max_frames)
            
            tasks.append(task)
            all_frames.append(frames)
            all_actions.append(actions_inbetween)

        return tasks, all_frames, all_actions

    def single_trajectory_loss(self, task_label, frames, actions):

        embedded_frames = self.representation_function.embed(torch.tensor(frames, device=self.device))
        vectorize_minerl_actions(actions, device=self.device)

        state = None
        for embedded_frame in embedded_frames:

            pass

        return 0


    def train_step(self):
        task_labels, batched_frames, batched_actions = self.get_batch()

        total_loss = 0
        for task_label, frames, actions in zip(task_labels, batched_frames, batched_actions):
            total_loss += self.single_trajectory_loss(task_label, frames, actions)

        print(len(frames), actions)

        for frames in batched_frames:

        self.representation_function.embed(frames)

        self.dynamics_function.forward()

        # TODO: load N trajectories (in the same way as XIRL did, with occasional frames with actions between them)

        # TODO: classify each trajectory as which task they belong to with a dynamics function

        # TODO: train the dynamics function to also 

        raise NotImplementedError


if __name__ == "__main__":
    tasks = ["MineRLBasaltBuildVillageHouse-v0"]

    cfg = FinalConfig(minerl_env_ids=tasks)
    trainer = FinalTrainer(cfg)
    trainer.train_step()