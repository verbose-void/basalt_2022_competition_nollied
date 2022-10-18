from dataclasses import asdict, dataclass, field
from typing import Callable

import gym
import torch
from torch.optim.lr_scheduler import StepLR
from fractal_zero.models.joint_model import JointModel

from fractal_zero.utils import get_space_shape


# DEFAULT_DEVICE = (
#     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
DEFAULT_DEVICE = torch.device("cpu")


CONSTANT_LR_CONFIG = {
    "alias": "ConstantLR",
    "class": StepLR,
    "step_size": 999999,
    "gamma": 1,
}


@dataclass
class FMCConfig:
    num_walkers: int = 8
    balance: float = 1

    track_game_tree: bool = True
    use_policy_for_action_selection: bool = False

    backprop_strategy: str = "all"  # all, clone_mask, or clone_participants
    clone_strategy: str = "cumulative_reward"  # cumulative_reward

    use_wandb: bool = False

    device = DEFAULT_DEVICE


@dataclass
class FractalZeroConfig:
    # TODO: break config into multiple parts (FMC, Trainer, etc.)

    env: gym.Env

    # TODO: if using AlphaZero style, autodetermine the embedding size.
    joint_model: JointModel

    fmc_config: FMCConfig = None

    max_replay_buffer_size: int = 512
    replay_buffer_pop_strategy: str = "oldest"  # oldest or random
    num_games: int = 5_000
    max_game_steps: int = 200

    max_batch_size: int = 128
    dynamic_batch_size: bool = True
    unroll_steps: int = 16
    minimize_batch_padding: bool = True
    learning_rate: float = 0.001
    lr_scheduler_config: dict = field(default_factory=lambda: CONSTANT_LR_CONFIG)
    weight_decay: float = 1e-4
    momentum: float = 0.9  # only if optimizer is SGD
    optimizer: str = "SGD"

    lookahead_steps: int = 64
    evaluation_lookahead_steps: int = 64

    device: torch.device = DEFAULT_DEVICE

    wandb_config: dict = None

    @property
    def use_wandb(self) -> bool:
        return self.wandb_config is not None

    @property
    def observation_shape(self) -> tuple:
        return get_space_shape(self.env.observation_space)

    @property
    def action_shape(self) -> tuple:
        return get_space_shape(self.env.action_space)

    def asdict(self) -> dict:
        d = asdict(self)

        del d["env"]
        d["env"] = self.env.unwrapped.spec.id

        return d
