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

        # self.dynamics_model.train()  # TODO

        num_sub_frames = len(embedded_sub_trajectory)
        assert num_sub_frames - 1 == len(actions_preceeding_each_timestep), f"{embedded_sub_trajectory.shape} and {len(actions_preceeding_each_timestep)}"

    def eval_step(self):
        # self.dynamics_model.eval()  # TODO

        pass
        