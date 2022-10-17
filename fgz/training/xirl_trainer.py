
from re import I
from typing import List
import ray
from vpt.agent import MineRLAgent
from fgz.architecture.dynamics_function import DynamicsFunction

from fgz.data_utils.xirl_data import XIRLDataHandler
from vpt.run_agent import load_agent

import torch


# @ray.remote
class XIRLTrainer:

    def __init__(self, dataset_path: str, model_path: str, weights_path: str):
        # NOTE: we can't use the same agent without more complicated thread-safeness code.
        self.agent = load_agent(model_path, weights_path)

        self.dynamics_function = DynamicsFunction(embedder_layers=2)

        self.data_handler = XIRLDataHandler(dataset_path, self.agent, self.dynamics_function)

    def soft_nearest_neighbor(self, frame_embedding: torch.Tensor, other_embeddings: List[torch.Tensor], return_similarity_vector: bool):
        # TODO: optimize this!
        soft_v = 0

        # similarity_vector = torch.zeros(size=(len(other_embeddings),), dtype=float)

        expanded_frame_embedding = frame_embedding.unsqueeze(0).expand(len(other_embeddings), -1)

        exp_norm = torch.exp(-torch.norm(expanded_frame_embedding - other_embeddings, dim=1))
        alpha_k = exp_norm / exp_norm.sum()

        if return_similarity_vector:
            return alpha_k

        return torch.matmul(alpha_k, other_embeddings)

    def train_on_pair(self, num_frames: int=20, regularization_weight: float=0.001):
        self.t0, self.t1 = self.data_handler.sample_pair()
        # self.t0 = torch.ones((10, 5))
        # self.t1 = torch.ones((10, 5))
        print("bytes for the pair of trajectories:", self.get_nbytes_stored())

        for _ in range(num_frames):
            # pick random frame in t0
            chosen_frame_index = torch.randint(low=0, high=len(self.t0), size=(1,)).item()
            ui = self.t0[chosen_frame_index]

            # calculate cycle MSE loss
            v_squiggly = self.soft_nearest_neighbor(ui, self.t1, return_similarity_vector=False)
            beta = self.soft_nearest_neighbor(v_squiggly, self.t0, return_similarity_vector=True)

            # TODO: vectorize
            mu = 0
            for i, beta_k in enumerate(beta):
                mu += beta_k * (i + 1)  # NOTE: i think + 1.

            variance = 0
            for i, beta_k in enumerate(beta):
                variance += beta_k * ((i + 1 - mu) ** 2)  # NOTE: i think + 1.

            loss = (torch.abs(chosen_frame_index - mu) ** 2) / variance
            std = torch.sqrt(variance)  # paper calls this variance, even though it's standard deviation.
            reg_term = regularization_weight * torch.log(std)

            loss += reg_term

            print(loss)
            loss.backward()

        # TODO: cycle consistency loss + train

    def get_nbytes_stored(self):
        nbytes0 = sum([e.nelement() * e.element_size() for e in self.t0])
        nbytes1 = sum([e.nelement() * e.element_size() for e in self.t1])
        return nbytes0 + nbytes1

