from typing import Tuple
import gym
import torch
import numpy as np
from fractal_zero.config import FractalZeroConfig
from fractal_zero.data.replay_buffer import ReplayBuffer
from fractal_zero.utils import get_space_shape


class DataHandler:
    def __init__(self, config: FractalZeroConfig):
        self.config = config

        self.replay_buffer = ReplayBuffer(self.config)

        # TODO: expert dataset

    def get_batch(
        self, num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: a version of this that allows non-uniform numbers of frames per batch

        assert num_frames > 0

        if self.config.dynamic_batch_size:
            # if using dynamic batch size, the batch size will never be greater than the number of
            # game histories in the replay buffer.
            batch_size = min(len(self.replay_buffer), self.config.max_batch_size)
        else:
            batch_size = self.config.max_batch_size

        observations = np.zeros(
            (batch_size, num_frames, *self.config.observation_shape), dtype=float
        )
        actions = np.zeros(
            (batch_size, num_frames, *self.config.action_shape), dtype=float
        )
        auxiliaries = np.zeros(
            (
                batch_size,
                num_frames,
            ),
            dtype=float,
        )
        values = np.zeros(
            (
                batch_size,
                num_frames,
            ),
            dtype=float,
        )

        total_empty_frames = 0
        for i in range(batch_size):
            (
                gobservations,
                gactions,
                grewards,
                gvalues,
                num_empty_frames,
            ) = self.replay_buffer.sample_game_clip(num_frames, pad_to_num_frames=True)

            observations[i, :] = gobservations
            actions[i] = np.expand_dims(
                gactions, -1
            )  # TODO: fix action shape to avoid this expanddims
            auxiliaries[i] = grewards  # auxiliary is a generalization of reward.
            values[i] = gvalues

            total_empty_frames += num_empty_frames

        # TODO: put these on the correct device sooner?
        return (
            torch.tensor(observations, device=self.config.device).float(),
            torch.tensor(actions, device=self.config.device).float(),
            torch.tensor(auxiliaries, device=self.config.device).unsqueeze(-1).float(),
            torch.tensor(values, device=self.config.device).unsqueeze(-1).float(),
            total_empty_frames,
        )
