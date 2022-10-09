import minerl
import gym

from typing import Dict, List

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

class DynamicsFunction(torch.nn.Module):
    def __init__(
        self, 
        state_embedding_size: int=1024, 
        button_features: int=16, 
        camera_features: int=16, 
        embedder_layers: int=4,
        discriminator_classes: int=2,
    ):
        super().__init__()

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

    def dummy_initial_state(self):
        return torch.zeros(self.state_embedding_size, dtype=float, requires_grad=True)

    def forward(self, state_embedding, buttons_vector, camera_vector):
        assert state_embedding.dim() <= 2 and state_embedding.shape[-1] == self.state_embedding_size, str(state_embedding.shape)

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


class MineRLDynamicsEnvironment(VectorizedEnvironment):

    def __init__(
        self, 
        action_space: gym.Env, 
        dynamics_function: DynamicsFunction, 
        n: int=1,
        apply_softmax_before_reward: bool = True,
    ):
        self.action_space = action_space
        self.dynamics_function = dynamics_function
        self.n = n

        # NOTE: this should be updated with each trajectory in the training script.
        self.target_discriminator_logit = None

        self.apply_softmax_before_reward = apply_softmax_before_reward

        self.states = dynamics_function.dummy_initial_state()

    def set_target_logit(self, target_logit: int):
        self.target_discriminator_logit = target_logit

    def set_all_states(self, state_embedding: torch.Tensor):
        assert state_embedding.dim() == 1, state_embedding.shape
        self.states = torch.zeros((self.n, self.dynamics_function.state_embedding_size))
        self.states[:] = state_embedding.cpu()

    def batch_step(self, actions, freeze_mask):
        assert len(actions) == self.n
        assert self.states.shape == (self.n, self.dynamics_function.state_embedding_size)

        button_vectors, camera_vectors = vectorize_minerl_actions(actions)
        new_states, discrim_logits = self.dynamics_function.forward(self.states, button_vectors, camera_vectors)

        # don't forward frozen states, frozen state's confusions are 0.
        self.states[freeze_mask != 1] = new_states[freeze_mask != 1]

        # applying a softmax before calculating the confusion reward means all of the
        # other logits are beingn minimized, including the FMC logit. this is ideal,
        # because it means the actions preferred by FMC simultaneously maximize the target logit
        # and minimize all of the others.
        if self.apply_softmax_before_reward:
            soft_logits = F.softmax(discrim_logits, dim=-1)
            soft_confusions = 1-soft_logits[:, self.target_discriminator_logit]
            rewards = torch.where(freeze_mask, 0, soft_confusions)
        else:
            target_discrim_class_confusions = 1-discrim_logits[:, self.target_discriminator_logit]
            rewards = torch.where(freeze_mask, 0, target_discrim_class_confusions)

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
            actions.append(action_space.sample())
        return actions