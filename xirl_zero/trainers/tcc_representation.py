from typing import Tuple
import torch

class TCCRepresentationTrainer:

    def train_step(self, t0: torch.Tensor, t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The representation function is trained using Temporal Cycle-Consistency (https://arxiv.org/pdf/1904.07846.pdf) 
        loss over a subset of the frames in a pair of demonstration trajectories.
        """

        return None, None  # TODO: return embedded