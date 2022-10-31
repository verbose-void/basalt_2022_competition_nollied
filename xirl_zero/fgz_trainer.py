

from dataclasses import dataclass
import os
import numpy as np

from typing import Dict, List

from fgz.architecture._dynamics_function import DynamicsFunction
from xirl_zero.data_utils.contiguous_trajectory_loader import ContiguousTrajectoryLoader


MINERL_DATA_ROOT = os.getenv("MINERL_DATA_ROOT", "data/")
VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")


@dataclass
class Config:
    minerl_env_ids: List[str]

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

    # representation_config: TCCConfig = field(default_factory=TCCConfig)
    # dynamics_config: MuZeroDynamicsConfig = field(default_factory=MuZeroDynamicsConfig)

    # def asdict(self):
    #     d = dict(self.__dict__)

    #     d.pop("representation_config")
    #     d.pop("dynamics_config")
    #     d["representation_config"] = self.representation_config.__dict__
    #     d["dynamics_config"] = self.dynamics_config.__dict__

    #     return d


class Trainer:

    def __init__(self, config: Config):
        self.config = config

        self.dynamics_discriminator = DynamicsFunction(2048, )

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

    def train_step(self):
        tasks, frames, actions = self.get_batch()
        print(len(frames), actions)

        # TODO: load N trajectories (in the same way as XIRL did, with occasional frames with actions between them)

        # TODO: classify each trajectory as which task they belong to with a dynamics function

        # TODO: train the dynamics function to also 

        raise NotImplementedError


if __name__ == "__main__":
    tasks = ["MineRLBasaltBuildVillageHouse-v0"]

    cfg = Config(minerl_env_ids=tasks)
    trainer = Trainer(cfg)
    trainer.train_step()