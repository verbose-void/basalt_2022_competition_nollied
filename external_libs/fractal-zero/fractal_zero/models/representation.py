import torch
import numpy as np
import gym

from fractal_zero.utils import get_space_shape


class FullyConnectedRepresentationModel(torch.nn.Module):
    def __init__(self, env: gym.Env, embedding_size: int):
        super().__init__()

        self.embedding_size = embedding_size
        self.observation_shape = get_space_shape(env.observation_space)

        in_dim = np.prod(self.observation_shape).astype(int)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, observation):
        return self.net(observation)
