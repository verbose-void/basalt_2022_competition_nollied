from dataclasses import dataclass, field
from typing import Dict
from gym import os
import torch

from datetime import datetime

import wandb

from xirl_zero.data_utils.contiguous_trajectory_loader import ContiguousTrajectoryLoader
from xirl_zero.trainers.tcc_representation import TCCConfig, TCCRepresentationTrainer
from xirl_zero.trainers.muzero_dynamics import MuZeroDynamicsConfig, MuZeroDynamicsTrainer

from xirl_zero.trainers.tcc_representation import MINERL_DATA_ROOT, VPT_MODELS_ROOT


@dataclass
class Config:
    dataset_dir: str

    train_steps: int
    eval_every: int
    eval_steps: int
    checkpoint_every: int

    num_frame_samples: int = 128

    verbose: bool = True

    use_wandb: bool = False

    # used for smoke tests
    max_frames: int = None
    max_trajectories: int = None
    model_log_frequency: int = 1000

    representation_config: TCCConfig = field(default_factory=TCCConfig)
    dynamics_config: MuZeroDynamicsConfig = field(default_factory=MuZeroDynamicsConfig)

    @property
    def dataset_path(self):
        return os.path.join(MINERL_DATA_ROOT, self.dataset_dir)

    def asdict(self):
        d = dict(self.__dict__)

        d.pop("representation_config")
        d.pop("dynamics_config")
        d["representation_config"] = self.representation_config.__dict__
        d["dynamics_config"] = self.dynamics_config.__dict__

        return d



class Trainer:
    """This trainer class should concurrently run 2 different trainers:
    
    -   One of them is the XIRL trainer, which is responsible for training the representation function
        inside of a structured embedding space, such that it can be used to guide the model
        with the XIRL reward (distance to the aggregate target embedding).

    -   The other is the dynamics function trainer, which is responsible for training the dynamics function
        to, given a sequence of actions and recurrent embeddings, ensure the embeddings correspond with the
        ground truth embeddings (from the XIRL embedder).

    Comments:
    -   Since we sample 2 trajectories for the TCC loss, we can have the dynamics function train on both of these
        sets of actions and embeddings.
    -   It's also nice, because by having the dynamics function learn a mapping between emebdding + action -> future
        embedding, it's essentially performing a sort of knowledge distillation, so we can compress the embedding and
        have it be predictive.

    Questions:
    -   Should these be steps in the training process or happen concurrently?
    """

    def __init__(self, config: Config):
        # each trainer instance belongs to only 1 task.

        self.config = config

        self.representation_trainer = TCCRepresentationTrainer(config.representation_config)
        self.dynamics_trainer = MuZeroDynamicsTrainer(config.dynamics_config)

        self.train_loader, self.eval_loader = ContiguousTrajectoryLoader.get_train_and_eval_loaders(config.dataset_path, max_trajectories=self.config.max_trajectories)

        self.train_steps_taken = 0

        if self.config.use_wandb:
            self.run_name = wandb.run.name
            wandb.watch(
                (self.representation_trainer.model, self.dynamics_trainer.model),
                log_freq=self.config.model_log_frequency,
            )

        else:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    def sample(self, from_train: bool):
        # TODO: latency hide dataloading?

        loader = self.train_loader if from_train else self.eval_loader

        kwargs = {"num_frame_samples": self.config.num_frame_samples, "max_frames": self.config.max_frames}
        t0, t0_actions = loader.sample(**kwargs)
        t1, t1_actions = loader.sample(**kwargs)

        return t0, t0_actions, t1, t1_actions

    def _get_data_stats(self, t0, t0_actions, t1, t1_actions):

        atotal = 0
        atotal += sum([len(actions) for actions in t0_actions])
        atotal += sum([len(actions) for actions in t1_actions])
        
        return {
            "total_frames": len(t0) + len(t1),
            "total_actions": atotal,
        }

    def train_step(self):
        t0, t0_actions, t1, t1_actions = self.sample(from_train=True)

        # train the representation function on it's own
        # TODO: should we give the representation function a head-start?
        tcc_stats, embedded_t0, embedded_t1 = self.representation_trainer.train_step(t0, t1)

        # with the representation function's outputs, train the dyanmics function to lookahead
        # TODO: should we latency hide the dynamics function training?
        zero_stats = self.dynamics_trainer.train_step(embedded_t0, t0_actions, embedded_t1, t1_actions)

        data_stats = self._get_data_stats(t0, t0_actions, t1, t1_actions)
        self.log(is_train=True, data_stats=data_stats, tcc_stats=tcc_stats, zero_stats=zero_stats)

        self.train_steps_taken += 1

    @torch.no_grad()
    def eval_step(self):
        t0, t0_actions, t1, t1_actions = self.sample(from_train=False)

        tcc_stats, embedded_t0, embedded_t1 = self.representation_trainer.eval_step(t0, t1)
        zero_stats = self.dynamics_trainer.eval_step(embedded_t0, t0_actions, embedded_t1, t1_actions)

        data_stats = self._get_data_stats(t0, t0_actions, t1, t1_actions)
        self.log(is_train=False, data_stats=data_stats, tcc_stats=tcc_stats, zero_stats=zero_stats)

    def get_target_state(self):
        # TODO: use both train and eval data to generate the target state.
        return self.representation_trainer.generate_target_state(self.train_loader)

    def log(self, is_train: bool, data_stats: Dict, tcc_stats: Dict, zero_stats: Dict):
        if self.config.verbose:
            if is_train:
                print("\n\n---------------- TRAIN ----------------")
            else:
                print("\n\n---------------- EVAL ----------------")

            print("TCC Stats:")
            print(tcc_stats)
            print("\nDynamics Function Stats:")
            print(zero_stats)
            print("\nData Stats:")
            print(data_stats)

        if self.config.use_wandb and wandb.run is not None:
            key = "train" if is_train else "eval"

            wandb.log({
                "train_steps": self.train_steps_taken,
                f"{key}/representation/": tcc_stats,
                f"{key}/dynamics/": zero_stats,
                f"{key}/data/": data_stats,
            })

    def checkpoint(self, directory: str):
        directory = os.path.join(directory, self.run_name, "checkpoints")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.train_steps_taken}.pth")

        # these variables are not saveable, so remove them before saving, then restore.
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        self.train_loader = None
        self.eval_loader = None

        torch.save(self, path)

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        return path

    def generate_and_save_target_state(self, directory: str):
        directory = os.path.join(directory, self.run_name, "target_states")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.train_steps_taken}.pth")
        target_state = self.get_target_state().cpu()
        torch.save(target_state, path)
        return path, target_state
