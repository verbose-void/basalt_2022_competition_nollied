



import minerl
import torch
import torch.nn.functional as F
import gym

import numpy as np

from xirl_zero.architecture.dynamics_function import DynamicsFunction, vectorize_minerl_actions



def _relativize_vector(vector: torch.Tensor):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard



class DynamicsFMC:

    def __init__(
        self, 
        dynamics_function: DynamicsFunction,
        target_state: torch.Tensor,
        action_space: gym.Space,
        num_walkers: int=32,
        steps: int=128,
        balance: float=1.0,
    ):
        self.dynamics_function = dynamics_function
        self.action_space = action_space
        self.target_state = target_state

        self.num_walkers = num_walkers
        self.steps = steps
        self.balance = balance

        self.step = None
        self.device = None
        self.states = None

        self.action_history = np.empty(shape=(steps, num_walkers), dtype=object)

    def sample_actions(self):
        raw_actions = [self.action_space.sample() for _ in range(self.num_walkers)]

        self.action_history[self.step, :] = raw_actions

        button_vecs, camera_vecs = vectorize_minerl_actions(raw_actions, device=self.device)
        return button_vecs, camera_vecs

    def get_actions(self, state_embedding: torch.Tensor):
        assert state_embedding.shape == (self.dynamics_function.state_embedding_size,)

        self.device = state_embedding.device
        self.states = torch.zeros(
            (self.num_walkers, self.dynamics_function.state_embedding_size),
            device=self.device,
        )

        self._simulate()

        raise NotImplementedError

    def _perturbate(self):
        button_vecs, camera_vecs = self.sample_actions()
        new_states = self.dynamics_function.forward(self.states, button_vecs, camera_vecs)

        print(new_states.shape)

    def _clone(self):
        # scores are inverse l2 distance to the target state.
        scores = -torch.norm(self.states[:] - self.target_state, dim=1)

        partners = torch.randint(low=0, high=self.num_walkers, size=(self.num_walkers,), device=self.device)
        walker_distances = 1 - F.cosine_similarity(self.states, self.states[partners], dim=-1)

        rel_scores = _relativize_vector(scores)
        rel_distances = _relativize_vector(walker_distances)

        virtual_rewards = rel_scores ** self.balance * rel_distances
        print(virtual_rewards)

    def _simulate(self):
        for self.step in range(self.steps):
            self._perturbate()
            self._clone()
    

if __name__ == "__main__":
    dynamics = DynamicsFunction()
    state = dynamics.dummy_initial_state()
    target_state = dynamics.dummy_initial_state()
    space = gym.make("MineRLBasaltFindCave-v0").action_space

    fmc = DynamicsFMC(dynamics, target_state, action_space=space, num_walkers=4, steps=16)
    fmc.get_actions(state)