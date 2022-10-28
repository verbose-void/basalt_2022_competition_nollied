import minerl
import gym

from typing import Dict, List
from vpt.agent import MineRLAgent

from vpt.lib.actions import Buttons
import torch
import torch.nn.functional as F

from fractal_zero.vectorized_environment import VectorizedEnvironment


num_actions = len(Buttons.ALL)


def _vectorize_buttons(action: Dict):
    a = torch.zeros(num_actions, dtype=float)
    for i, button_str in enumerate(Buttons.ALL):
        a[i] = torch.tensor(action[button_str], dtype=float)
    a[a > 0] = torch.softmax(a[a > 0], dim=0)
    return a


def vectorize_minerl_action(action: Dict, camera_scale: float = 180):
    # group movement keys
    button_vec = _vectorize_buttons(action)
    camera_vec = torch.tensor(action["camera"], dtype=float) / camera_scale
    return button_vec.float(), camera_vec.squeeze().float()


def vectorize_minerl_actions(actions: List[Dict], camera_scale: float = 180, device=None):
    button_vecs = []
    camera_vecs = []
    for action in actions:
        bv, cv = vectorize_minerl_action(action, camera_scale=camera_scale)
        button_vecs.append(bv)
        camera_vecs.append(cv)

    button_vecs, camera_vecs = torch.stack(button_vecs), torch.stack(camera_vecs)

    if device is not None:
        return button_vecs.to(device), camera_vecs.to(device)

    return button_vecs, camera_vecs


class DynamicsFunction(torch.nn.Module):
    def __init__(
        self,
        state_embedding_size: int = 1024,
        button_features: int = 16,
        camera_features: int = 16,
        embedder_layers: int = 4,
    ):
        super().__init__()

        self.state_embedding_size = state_embedding_size

        self.button_embedder = torch.nn.Sequential(
            torch.nn.Linear(num_actions, button_features), torch.nn.ReLU()
        )

        self.camera_embedder = torch.nn.Sequential(
            torch.nn.Linear(2, camera_features), torch.nn.ReLU()
        )

        embeds = [
            torch.nn.Linear(
                state_embedding_size + button_features + camera_features,
                state_embedding_size,
            )
        ]
        for _ in range(embedder_layers):
            embeds.extend(
                (
                    torch.nn.ReLU(),
                    torch.nn.Linear(state_embedding_size, state_embedding_size),
                )
            )
        self.embedder = torch.nn.Sequential(
            *embeds,
            # torch.nn.Sigmoid(),  # prevent discriminator from exploiting FMC similarity measure.
        )

    def dummy_initial_state(self):
        return torch.zeros(self.state_embedding_size, dtype=float, requires_grad=True)

    def forward(
        self, state_embedding, buttons_vector, camera_vector,
    ):
        assert (
            state_embedding.dim() <= 2
            and state_embedding.shape[-1] == self.state_embedding_size
        ), (str(state_embedding.shape), self.state_embedding_size)

        button_embedding = self.button_embedder.forward(buttons_vector)
        camera_embedding = self.camera_embedder.forward(camera_vector)

        concat_state = torch.cat(
            (state_embedding, button_embedding, camera_embedding), dim=-1
        )
        new_state = self.embedder.forward(concat_state)

        return new_state

    def forward_action(
        self, state_embedding: torch.Tensor, action, use_discrim: bool = True
    ):
        button_vec, camera_vec = vectorize_minerl_action(action)

        device = state_embedding.device
        button_vec = button_vec.unsqueeze(0).to(device)
        camera_vec = camera_vec.unsqueeze(0).to(device)

        return self.forward(
            state_embedding, button_vec, camera_vec, use_discrim=use_discrim
        )


class MineRLDynamicsEnvironment(VectorizedEnvironment):
    def __init__(
        self,
        action_space: gym.Env,
        dynamics_function: DynamicsFunction,
        target_state: torch.Tensor,
        n: int = 1,
    ):
        self.action_space = action_space
        self.dynamics_function = dynamics_function
        self.target_state = target_state
        self.n = n

        # NOTE: this should be updated with each trajectory in the training script.
        self.target_discriminator_logit = None

        self.states = dynamics_function.dummy_initial_state()

    def set_target_logit(self, target_logit: int):
        self.target_discriminator_logit = target_logit

    def set_all_states(self, state_embedding: torch.Tensor):
        assert state_embedding.dim() == 1, state_embedding.shape
        self.states = torch.zeros((self.n, self.dynamics_function.state_embedding_size), device=state_embedding.device)
        self.states[:] = state_embedding

    def batch_step(self, actions, freeze_mask):
        freeze_mask = freeze_mask.to(self.states.device)

        self.dynamics_function.eval()

        assert len(actions) == self.n
        assert self.states.shape == (
            self.n,
            self.dynamics_function.state_embedding_size,
        )

        button_vectors, camera_vectors = vectorize_minerl_actions(actions, device=self.states.device)
        new_states = self.dynamics_function.forward(
            self.states, button_vectors, camera_vectors
        )

        # don't forward frozen states, frozen state's confusions are 0.
        self.states[freeze_mask != 1] = new_states[freeze_mask != 1]

        raise NotImplementedError("TODO: inverse distance reward")

        obs = self.states
        dones = torch.zeros(self.n).bool()
        # infos = [{} for _ in range(len(self.states))]

        infos = discrim_logits

        return self.states, obs, rewards, dones, infos

    def clone(self, partners, clone_mask):
        self.states[clone_mask] = self.states[partners[clone_mask]]

    def batch_reset(self):
        # no need to be able to reset for our purposes.
        return self.states

    def batched_action_space_sample(self):
        actions = []
        for _ in range(self.n):
            action_space = self.action_space
            action = action_space.sample()
            actions.append(action)
        return actions
