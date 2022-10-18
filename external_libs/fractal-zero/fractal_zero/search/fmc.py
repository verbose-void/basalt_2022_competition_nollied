from copy import copy, deepcopy
from typing import Callable, List
import torch
import numpy as np

from tqdm import tqdm
from fractal_zero.search.tree import GameTree
from fractal_zero.utils import cloning_primitive

from fractal_zero.vectorized_environment import VectorizedEnvironment


def _l2_distance(vec0, vec1):
    if vec0.dim() > 2:
        vec0 = vec0.flatten(start_dim=1)
    elif vec0.dim() == 1:
        vec0 = vec0.unsqueeze(-1)
    if vec1.dim() > 2:
        vec1 = vec1.flatten(start_dim=1)
    elif vec1.dim() == 1:
        vec1 = vec1.unsqueeze(-1)
    return torch.norm(vec0 - vec1, dim=-1)


def _relativize_vector(vector: torch.Tensor):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


_ATTRIBUTES_TO_CLONE = (
    "states",
    "observations",
    "rewards",
    "dones",
    "scores",
    "average_rewards",
    "actions",
    "infos",
)


class FMC:
    def __init__(
        self,
        vectorized_environment: VectorizedEnvironment,
        balance: float = 1.0,
        disable_cloning: bool = False,
        use_average_rewards: bool = False,
        similarity_function: Callable = _l2_distance,
        freeze_best: bool = True,
        track_tree: bool = True,
        prune_tree: bool = True,
    ):
        self.vec_env = vectorized_environment
        self.balance = balance
        self.disable_cloning = disable_cloning
        self.use_average_rewards = use_average_rewards
        self.similarity_function = similarity_function

        self.freeze_best = freeze_best
        self.track_tree = track_tree
        self.prune_tree = prune_tree

        self.reset()

    def reset(self):
        # TODO: may need to make this decision of root observations more effectively for stochastic environments.
        self.observations = self.vec_env.batch_reset()
        root_obs = self.observations[0]

        self.dones = torch.zeros(self.num_walkers).bool()
        self.states, self.observations, self.rewards, self.infos = (
            None,
            None,
            None,
            None,
        )

        self.scores = torch.zeros(self.num_walkers, dtype=float)
        self.average_rewards = torch.zeros(self.num_walkers, dtype=float)
        self.clone_mask = torch.zeros(self.num_walkers, dtype=bool)
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)

        self.tree = (
            GameTree(self.num_walkers, prune=self.prune_tree, root_observation=root_obs)
            if self.track_tree
            else None
        )
        self.did_early_exit = False

    @property
    def num_walkers(self):
        return self.vec_env.n

    def _can_early_exit(self):
        return torch.all(self.dones)

    def simulate(self, steps: int, use_tqdm: bool = False):
        if self.did_early_exit:
            raise ValueError("Already early exited.")

        it = tqdm(range(steps), disable=not use_tqdm)
        for _ in it:
            self._perturbate()

            if self._can_early_exit():
                self.did_early_exit = True
                break

            self._clone()

    def _perturbate(self):
        freeze_steps = torch.logical_or(self.freeze_mask, self.dones)

        # TODO: don't sample actions for frozen environments? (make sure to remove the comments about this)
        # will make it more legible.
        self.actions = self.vec_env.batched_action_space_sample()

        (
            self.states,
            self.observations,
            self.rewards,
            self.dones,
            self.infos,
        ) = self.vec_env.batch_step(self.actions, freeze_steps)
        self.scores += self.rewards
        self.average_rewards = self.scores / self.tree.get_depths()

        if self.tree:
            # NOTE: the actions that are in the tree will diverge slightly from
            # those being represented by FMC's internal state. The reason for this is
            # that when we freeze certain environments in the vectorized environment
            # object, the actions that are sampled will not be enacted, and the previous
            # return values will be provided.
            self.tree.build_next_level(
                self.actions,
                self.observations,
                self.rewards,
                self.infos,
                freeze_steps,
            )

        self._set_freeze_mask()

    def _set_freeze_mask(self):
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)
        if self.freeze_best:
            if self.use_average_rewards:
                metric = self.average_rewards.argmax()
            else:
                metric = self.scores
            self.freeze_mask[metric.argmax()] = 1

    def _set_valid_clone_partners(self):
        valid_clone_partners = np.arange(self.num_walkers)

        # cannot clone to walkers at terminal states
        valid_clone_partners = valid_clone_partners[(self.dones == False).numpy()]

        # TODO: make it so walkers cannot clone to themselves
        clone_partners = np.random.choice(valid_clone_partners, size=self.num_walkers)
        self.clone_partners = torch.tensor(clone_partners, dtype=int).long()

    def _set_clone_variables(self):
        self._set_valid_clone_partners()
        self.similarities = self.similarity_function(
            self.states, self.states[self.clone_partners]
        )

        rel_sim = _relativize_vector(self.similarities)

        if self.use_average_rewards:
            rel_score = _relativize_vector(self.average_rewards)
        else:
            rel_score = _relativize_vector(self.scores)

        self.virtual_rewards = rel_score**self.balance * rel_sim

        vr = self.virtual_rewards
        pair_vr = self.virtual_rewards[self.clone_partners]
        value = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        self.clone_mask = (value >= torch.rand(1)).bool()

        # clone all walkers at terminal states
        self.clone_mask[
            self.dones
        ] = True  # NOTE: sometimes done might be a preferable terminal state (winning)... deal with this.
        # don't clone frozen walkers
        self.clone_mask[self.freeze_mask] = False

    def _clone(self):
        self._set_clone_variables()

        if self.disable_cloning:
            return
        self.vec_env.clone(self.clone_partners, self.clone_mask)
        if self.tree:
            self.tree.clone(self.clone_partners, self.clone_mask)

        # doing this allows the GameTree to retain gradients in case training a model on FMC outputs.
        # it also is required to keep the cloning mechanism in-tact (because of inplace updates).
        self.rewards = self.rewards.clone()
        if isinstance(self.observations, torch.Tensor):
            self.observations = self.observations.clone()
        if isinstance(self.infos, torch.Tensor):
            self.infos = self.infos.clone()

        for attr in _ATTRIBUTES_TO_CLONE:
            self._clone_variable(attr)

        # sanity checks (TODO: maybe remove this?)
        if not torch.allclose(self.scores, self.tree.get_total_rewards(), rtol=0.001):
            raise ValueError(self.scores, self.tree.get_total_rewards())
        # if self.rewards[self.freeze_mask].sum().item() != 0:
        #     raise ValueError(self.rewards[self.freeze_mask], self.rewards[self.freeze_mask].sum())

    def _clone_variable(self, subject_var_name: str):
        subject = getattr(self, subject_var_name)
        # note: this will be cloned in-place!
        cloned_subject = cloning_primitive(
            subject, self.clone_partners, self.clone_mask
        )
        setattr(self, subject_var_name, cloned_subject)
        return cloned_subject
