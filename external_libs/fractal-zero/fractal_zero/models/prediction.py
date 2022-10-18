import torch
import torch.nn.functional as F

import gym
import numpy as np

from fractal_zero.utils import get_space_shape


class FullyConnectedPredictionModel(torch.nn.Module):
    def __init__(self, env: gym.Env, embedding_size: int):
        super().__init__()

        self.action_shape = get_space_shape(env.action_space)

        self.policy_head = torch.nn.Linear(embedding_size, np.prod(self.action_shape))
        self.value_head = torch.nn.Linear(embedding_size, 1)

        self.policy_loss = None  # TODO
        self.value_loss = F.cross_entropy

    def forward(self, embedding, with_randomness: bool = False):
        policy_logits = self.policy_head(
            embedding
        )  # TODO: wtf happens with continuous action spaces?
        value_prediction = self.value_head(embedding)

        return policy_logits, value_prediction
