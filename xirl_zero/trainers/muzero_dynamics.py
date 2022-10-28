from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from fgz.architecture.dynamics_function import DynamicsFunction


@dataclass
class MuZeroDynamicsConfig:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MuZeroDynamicsTrainer:

    def __init__(self):
        self.config = MuZeroDynamicsConfig()

        # TODO: use pretrained weights/steal architecture from the agent!
        self.model = DynamicsFunction(2048, button_features=32, camera_features=32).to(self.config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

    def calculate_loss(
        self, 
        embedded_sub_trajectory: torch.Tensor, 
        actions_preceeding_each_timestep: Sequence,
    ):
        num_sub_frames = len(embedded_sub_trajectory)
        assert num_sub_frames - 1 == len(actions_preceeding_each_timestep), f"{embedded_sub_trajectory.shape} and {len(actions_preceeding_each_timestep)}"

        total_loss = 0

        for i in range(num_sub_frames - 1):
            start_embedding = embedded_sub_trajectory[i]
            end_embedding = embedded_sub_trajectory[i + 1]
            actions_between = actions_preceeding_each_timestep[i]

            # TODO: there are many variations to how the target embedding can be determined.
            # target_embedding = (start_embedding + end_embedding) / 2
            target_embedding = end_embedding

            between_loss = 0

            embedding = start_embedding.unsqueeze(0).to(self.config.device)
            for action in actions_between:
                embedding = self.model.forward_action(embedding, action, use_discrim=False)
                between_loss += F.mse_loss(embedding, target_embedding)
            total_loss += between_loss / len(actions_between)

        loss = total_loss / num_sub_frames
        return loss

    def train_step(
        self, 
        t0: torch.Tensor, 
        a0: Sequence,
        t1: torch.Tensor, 
        a1: Sequence,
    ):
        """The dynamics function is trained to, given a subset of the trajectory embeddings (but all of the actions), 
        predict the following embedding. 
        """

        self.model.train()
        self.optimizer.zero_grad()

        loss0 = self.calculate_loss(t0, a0)
        loss1 = self.calculate_loss(t1, a1)
        loss = (loss0 + loss1) / 2
        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item,
        }

    @torch.no_grad()
    def eval_step(
        self, 
        t0: torch.Tensor, 
        a0: Sequence,
        t1: torch.Tensor, 
        a1: Sequence,):

        self.model.eval()

        loss0 = self.calculate_loss(t0, a0)
        loss1 = self.calculate_loss(t1, a1)
        loss = (loss0 + loss1) / 2

        return {
            "loss": loss,
        }
        