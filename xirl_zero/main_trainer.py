from dataclasses import dataclass
import torch
from xirl_zero.data_utils.contiguous_trajectory_loader import ContiguousTrajectoryLoader


from xirl_zero.trainers.tcc_representation import TCCConfig, TCCRepresentationTrainer
from xirl_zero.trainers.muzero_dynamics import MuZeroDynamicsTrainer


SMOKE_TEST = True


@dataclass
class Config:
    dataset_path: str

    num_frame_samples: int = 128

    # used for smoke tests
    max_frames: int = 10 if SMOKE_TEST else None

    verbose: bool = True


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

        self.representation_trainer = TCCRepresentationTrainer(TCCConfig())
        self.dynamics_trainer = MuZeroDynamicsTrainer()

        self.train_loader, self.eval_loader = ContiguousTrajectoryLoader.get_train_and_eval_loaders(config.dataset_path)

    def sample(self, from_train: bool):
        # TODO: latency hide dataloading?

        loader = self.train_loader if from_train else self.eval_loader

        kwargs = {"num_frame_samples": self.config.num_frame_samples, "max_frames": self.config.max_frames}
        t0, t0_actions = loader.sample(**kwargs)
        t1, t1_actions = loader.sample(**kwargs)

        return t0, t0_actions, t1, t1_actions

    def train_step(self):
        t0, t0_actions, t1, t1_actions = self.sample(from_train=True)

        # train the representation function on it's own
        # TODO: should we give the representation function a head-start?
        tcc_stats, embedded_t0, embedded_t1 = self.representation_trainer.train_step(t0, t1)

        # with the representation function's outputs, train the dyanmics function to lookahead
        self.dynamics_trainer.train_step(embedded_t0, t0_actions)
        self.dynamics_trainer.train_step(embedded_t1, t1_actions)

        if self.config.verbose:
            print("\n\n---------------- TRAIN ----------------")
            print("TCC Stats:")
            print(tcc_stats)
            print("Dynamics Function Stats:")
            print({})  # TODO

    @torch.no_grad()
    def eval_step(self):
        t0, t0_actions, t1, t1_actions = self.sample(from_train=False)

        tcc_stats, embedded_t0, embedded_t1 = self.representation_trainer.eval_step(t0, t1)

        # self.dynamics_trainer.eval_step(embedded_t0, t0_actions)
        # self.dynamics_trainer.eval_step(embedded_t1, t1_actions)

        if self.config.verbose:
            print("\n\n---------------- EVAL ----------------")
            print("TCC Stats:")
            print(tcc_stats)
            print("Dynamics Function Stats:")
            print({})  # TODO