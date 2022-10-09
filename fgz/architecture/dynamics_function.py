import minerl
import gym

from typing import Dict, List

from vpt.lib.actions import Buttons
import torch


num_actions = len(Buttons.ALL)

def _vectorize_buttons(action: Dict):
    a = torch.zeros(num_actions, dtype=float)
    for i, button_str in enumerate(Buttons.ALL):
        a[i] = torch.tensor(action[button_str], dtype=float)
    a[a > 0] = torch.softmax(a[a > 0], dim=0)
    return a

def vectorize_minerl_action(action: Dict, camera_scale: float=180):
    # group movement keys
    button_vec = _vectorize_buttons(action)
    camera_vec = torch.tensor(action["camera"], dtype=float) / camera_scale
    return button_vec.float(), camera_vec.float()

def vectorize_minerl_actions(actions: List[Dict], camera_scale: float=180):
    button_vecs = []
    camera_vecs = []
    for action in actions:
        bv, cv = vectorize_minerl_action(action, camera_scale=camera_scale)
        button_vecs.append(bv)
        camera_vecs.append(cv)
    return torch.stack(button_vecs), torch.stack(camera_vecs)

class DynamicsFunction:
    def __init__(
        self, 
        state_embedding_size: int=4096, 
        button_features: int=16, 
        camera_features: int=16, 
        embedder_layers: int=4,
        discriminator_classes: int=2,
    ):
        self.state_embedding_size = state_embedding_size

        self.button_embedder = torch.nn.Sequential(
            torch.nn.Linear(num_actions, button_features),
            torch.nn.ReLU(),
        )

        self.camera_embedder = torch.nn.Sequential(
            torch.nn.Linear(2, camera_features),
            torch.nn.ReLU(),
        )

        embeds = [
            torch.nn.Linear(state_embedding_size + button_features + camera_features, state_embedding_size)
        ]
        for _ in range(embedder_layers):
            embeds.extend((torch.nn.ReLU(), torch.nn.Linear(state_embedding_size, state_embedding_size)))
        self.embedder = torch.nn.Sequential(
            *embeds, 
            torch.nn.Sigmoid(),  # prevent discriminator from exploiting FMC similarity measure.
        )

        self.discriminator_head = torch.nn.Sequential(
            torch.nn.Linear(state_embedding_size, discriminator_classes),
            torch.nn.Softmax(),  # prevent discriminator from making FMC think it's getting high rewards when the scale is just large
        )

    def forward(self, state_embedding, buttons_vector, camera_vector):
        assert state_embedding.dim() <= 2 and state_embedding.shape[-1] == self.state_embedding_size

        button_embedding = self.button_embedder.forward(buttons_vector)
        camera_embedding = self.camera_embedder.forward(camera_vector)

        concat_state = torch.cat((state_embedding, button_embedding, camera_embedding), dim=-1)
        new_state = self.embedder.forward(concat_state)

        # TODO: residual from old state embedding to new?
        discriminator_logits = self.discriminator_head(new_state)

        return new_state, discriminator_logits

    def forward_action(self, state_embedding, action):
        button_vec, camera_vec = vectorize_minerl_action(action)
        # TODO: handle GPU
        return self.forward(state_embedding.cpu(), button_vec.unsqueeze(0), camera_vec.unsqueeze(0))


class MineRLDynamicsEnvironment:

    def __init__(
        self, 
        action_space: gym.Env, 
        dynamics_function: DynamicsFunction, 
        target_discriminator_logit: int, 
        n: int=1,
    ):
        self.action_space = action_space
        self.dynamics_function = dynamics_function
        self.n = n
        self.target_discriminator_logit = target_discriminator_logit

    def set_all_states(self, state_embedding: torch.Tensor):
        assert state_embedding.dim() == 1
        self.states = torch.zeros((self.n, self.dynamics_function.state_embedding_size))
        self.states[:] = state_embedding.cpu()

    def batch_step(self, actions):
        assert len(actions) == self.n
        assert self.states.shape == (self.n, self.dynamics_function.state_embedding_size)

        button_vectors, camera_vectors = vectorize_minerl_actions(actions)
        new_states, discrim_logits = self.dynamics_function.forward(self.states, button_vectors, camera_vectors)
        self.states = new_states

        target_discrim_class_confusions = 1-discrim_logits[:, self.target_discriminator_logit]

        obs = new_states
        reward = target_discrim_class_confusions
        done = False
        info = None

        return obs, reward, done, info