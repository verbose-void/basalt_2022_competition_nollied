from typing import Sequence
import torch


class MuZeroDynamicsTrainer:

    def train_step(
        self, 
        embedded_sub_trajectory: torch.Tensor, 
        actions_preceeding_each_timestep: Sequence,
    ):
        """The dynamics function is trained to, given a subset of the trajectory embeddings (but all of the actions), 
        predict the following embedding. 
        """

        num_sub_frames = len(embedded_sub_trajectory)
        assert num_sub_frames == len(actions_preceeding_each_timestep)