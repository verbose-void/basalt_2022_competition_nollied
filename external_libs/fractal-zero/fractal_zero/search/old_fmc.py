from copy import deepcopy
from typing import List, Union
from uuid import uuid4
from warnings import warn
import torch
import numpy as np
from tqdm import tqdm

import wandb
from fractal_zero.config import FMCConfig

from fractal_zero.search.tree import GameTree

from fractal_zero.utils import mean_min_max_dict
from fractal_zero.vectorized_environment import (
    VectorizedDynamicsModelEnvironment,
    VectorizedEnvironment,
)


@torch.no_grad()
def _relativize_vector(vector):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


class FMC:
    """Fractal Monte Carlo is a collaborative cellular automata based tree search algorithm. This version is special, because instead of having a gym
    environment maintain the state for each walker during the search process, each walker's state is represented inside of a batched hidden
    state variable inside of a dynamics model. Basically, the dynamics model's hidden state is of shape (num_walkers, *embedding_shape).

    This is inspired by Muzero's technique to have a dynamics model be learned such that the tree search need not interact with the environment
    itself. With FMC, it is much more natural than with MCTS, mostly because of the cloning phase being contrastive. As an added benefit of this
    approach, it's natively vectorized so it can be put onto the GPU.
    """

    vectorized_environment: VectorizedEnvironment

    def __init__(
        self,
        vectorized_environment: VectorizedEnvironment,
        config: FMCConfig = None,
        verbose: bool = False,
    ):
        self.vectorized_environment = vectorized_environment
        self.verbose = verbose

        self.config = self._build_default_config() if config is None else config
        self._validate_config()

        # TODO: maybe this reset and game tree construction should be called more cautiously.
        self.observations = self.vectorized_environment.batch_reset()

        if self.config.track_game_tree:
            root_observation = self.observations[0]
            self.tree = GameTree(self.num_walkers, root_observation)
        else:
            self.tree = None

        self.reset()

    def reset(self):
        # TODO: explain all these variables
        # NOTE: they should exist on the CPU.
        self.value_sum_buffer = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=float,
        )
        self.visit_buffer = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=int,
        )
        self.clone_receives = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=int,
        )
        self.cumulative_rewards = torch.zeros(size=(self.num_walkers, 1), dtype=float)
        self.root_actions = None

    def _build_default_config(self) -> FMCConfig:
        return FMCConfig(num_walkers=self.vectorized_environment.n)

    def _validate_num_walkers(self):
        if self.config.num_walkers != self.vectorized_environment.n:
            raise ValueError(
                f"Expected config num walkers ({self.config.num_walkers}) and vectorized environment n ({self.vectorized_environment.n}) to match."
            )

    def _validate_config(self):
        if self.config.use_policy_for_action_selection:
            raise NotImplementedError(
                "Using policy functions to sample walker actions not yet supported."
            )
        self._validate_num_walkers()

    @property
    def num_walkers(self) -> int:
        return self.config.num_walkers

    @property
    def device(self):
        return self.config.device

    @torch.no_grad()
    def _perturbate(self):
        """Advance the state of each walker."""

        self._assign_actions()

        (
            self.states,
            self.observations,
            self.rewards,
            self.dones,
            _,
        ) = self.vectorized_environment.batch_step(self.actions)

        self.cumulative_rewards += self.rewards

        if self.tree:
            self.tree.build_next_level(self.actions, self.observations, self.rewards)

    @torch.no_grad()
    def simulate(self, k: int, greedy_action: bool = True, use_tqdm: bool = False):
        """Run FMC for k iterations, returning the best action that was taken at the root/initial state."""

        self.k = k
        assert self.k > 0

        it = tqdm(
            range(self.k),
            desc="Simulating with FMC",
            total=self.k,
            disable=not use_tqdm,
        )
        for self.simulation_iteration in it:
            # in case the vectorized environment was used to perform another operation.
            self._validate_num_walkers()

            self._perturbate()
            self._prepare_clone_variables()
            self._execute_cloning()

        # TODO: try to convert the root action distribution into a policy distribution? this may get hard in continuous action spaces. https://arxiv.org/pdf/1805.09613.pdf

        self.log(
            {
                **mean_min_max_dict("fmc/visit_buffer", self.visit_buffer.float()),
                **mean_min_max_dict("fmc/value_sum_buffer", self.value_sum_buffer),
                **mean_min_max_dict(
                    "fmc/average_value_buffer",
                    self.value_sum_buffer / self.visit_buffer.float(),
                ),
                **mean_min_max_dict("fmc/clone_receives", self.clone_receives.float()),
            },
            commit=False,
        )

        return self.get_root_action(greedy_action)

    def get_root_action(self, greedy: bool):
        if greedy:
            return self._get_action_with_highest_cumulative_reward()
        return self._get_action_with_highest_clone_receives()

    @torch.no_grad()
    def _assign_actions(self):
        """Each walker picks an action to advance it's state."""

        self.actions = self.vectorized_environment.batched_action_space_sample()
        if self.root_actions is None:
            self.root_actions = deepcopy(self.actions)

    @torch.no_grad()
    def _assign_clone_partners(self):
        """For the cloning phase, walkers need a partner to determine if they should be sent as reinforcements to their partner's state."""

        choices = np.random.choice(np.arange(self.num_walkers), size=self.num_walkers)
        self.clone_partners = torch.tensor(choices, dtype=int)

    @torch.no_grad()
    def _calculate_distances(self):
        """For the cloning phase, we calculate the distances between each walker and their partner for balancing exploration."""

        self.distances = torch.linalg.norm(
            self.states - self.states[self.clone_partners], dim=1
        )

    @torch.no_grad()
    def _calculate_virtual_rewards(self):
        """For the cloning phase, we calculate a virtual reward that is the composite of each walker's distance to their partner weighted with
        their rewards. This is used to determine the probability to clone and is used to balance exploration and exploitation.

        Both the reward and distance vectors are "relativized". This keeps all of the values in each vector contextually scaled with each step.
        The authors of Fractal Monte Carlo claim this is a method of shaping a "universal reward function". Without relativization, the
        vectors may have drastically different ranges, causing more volatility in how many walkers are cloned at each step. If the reward or distance
        ranges were too high, it's likely no cloning would occur at all. If either were too small, then it's likely all walkers would be cloned.
        """

        # TODO EXPERIMENT: should we be using the value estimates? or should we be using the value buffer?
        # or should we be using the cumulative rewards? (the original FMC authors use cumulative rewards)
        clone_strat = self.config.clone_strategy
        if clone_strat == "cumulative_reward":
            exploit = self.cumulative_rewards
        else:
            raise ValueError(f"Clone strat {clone_strat} not supported.")

        rel_exploits = _relativize_vector(exploit).squeeze(-1).cpu()
        rel_distances = _relativize_vector(self.distances).cpu()
        self.virtual_rewards = (rel_exploits**self.config.balance) * rel_distances

        self.log(
            {
                **mean_min_max_dict("fmc/virtual_rewards", self.virtual_rewards),
                **mean_min_max_dict("fmc/distances", self.distances),
                **mean_min_max_dict("fmc/auxiliaries", self.rewards),
            },
            commit=False,
        )

    @torch.no_grad()
    def _determine_clone_receives(self):
        # keep track of which walkers received clones and how many.
        clones_received_per_walker = torch.bincount(
            self.clone_partners[self.clone_mask]
        ).unsqueeze(-1)

        n = len(clones_received_per_walker)

        self.clone_receives[:n] += clones_received_per_walker

        self.clone_receive_mask = torch.zeros_like(self.clone_mask)
        self.clone_receive_mask[:n] = clones_received_per_walker.squeeze(-1) > 0

    @torch.no_grad()
    def _determine_clone_mask(self):
        """The clone mask is based on the virtual rewards of each walker and their clone partner. If a walker is selected to clone, their
        state will be replaced with their partner's state.
        """

        vr = self.virtual_rewards
        pair_vr = vr[self.clone_partners]

        self.clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        r = np.random.uniform()
        self.clone_mask = (self.clone_probabilities >= r).cpu()

        self._determine_clone_receives()

        self.log(
            {
                "fmc/num_cloned": self.clone_mask.sum(),
            },
            commit=False,
        )

    @torch.no_grad()
    def _prepare_clone_variables(self):
        # TODO: docstring

        # prepare virtual rewards and partner virtual rewards
        self._assign_clone_partners()
        self._calculate_distances()
        self._calculate_virtual_rewards()
        self._determine_clone_mask()

    @torch.no_grad()
    def _execute_cloning(self):
        """The cloning phase is where the collaboration of the cellular automata comes from. Using the virtual rewards calculated for
        each walker and clone partners that are randomly assigned, there is a probability that some walkers will be sent as reinforcements
        to their randomly assigned clone partner.

        The goal of the clone phase is to maintain a balanced density over state occupations with respect to exploration and exploitation.
        """

        # TODO: don't clone best walker (?)
        # execute clones
        self.vectorized_environment.clone(self.clone_partners, self.clone_mask)

        self.states = self._clone(self.states)
        self.observations = self._clone(self.observations)
        self.rewards = self._clone(self.rewards)

        self.actions = self._clone(self.actions)
        self.root_actions = self._clone(self.root_actions)
        self.cumulative_rewards = self._clone(self.cumulative_rewards)
        self.visit_buffer = self._clone(self.visit_buffer)
        self.clone_receives = self._clone(
            self.clone_receives
        )  # yes... clone clone receives lol.

        if self.tree:
            self.tree.clone(self.clone_partners, self.clone_mask)

        if self.verbose:
            print("state after", self.state)

    @torch.no_grad()
    def _get_action_with_highest_cumulative_reward(self):
        # TODO: docstring

        walker_index = self.cumulative_rewards.argmax(0)
        return self.root_actions[walker_index]

    @torch.no_grad()
    def _get_action_with_highest_clone_receives(self):
        # TODO: docstring
        most_cloned_to_walker = torch.argmax(self.clone_receives)
        return self.root_actions[most_cloned_to_walker]

    def _clone(self, subject):
        if isinstance(subject, torch.Tensor):
            return self._clone_vector(subject)
        elif isinstance(subject, list):
            return self._clone_list(subject)
        raise NotImplementedError

    def _clone_vector(self, vector: torch.Tensor):
        vector[self.clone_mask] = vector[self.clone_partners[self.clone_mask]]
        return vector

    def _clone_list(self, l: List, copy: bool = False):
        new_list = []
        for i in range(self.num_walkers):
            do_clone = self.clone_mask[i]
            partner = self.clone_partners[i]

            if do_clone:
                # NOTE: may not need to deepcopy.
                if copy:
                    new_list.append(deepcopy(l[partner]))
                else:
                    new_list.append(l[partner])
            else:
                new_list.append(l[i])
        return new_list

    def _clone_actions(self):
        new_leaf_actions = []
        new_root_actions = []

        for i in range(self.num_walkers):
            do_clone = self.clone_mask[i]
            partner = self.clone_partners[i]

            if do_clone:
                # NOTE: may not need to deepcopy.
                new_leaf_actions.append(deepcopy(self.actions[partner]))
                new_root_actions.append(deepcopy(self.root_actions[partner]))
            else:
                new_leaf_actions.append(self.actions[i])
                new_root_actions.append(self.root_actions[i])

        self.actions = new_leaf_actions
        self.root_actions = new_root_actions

    def log(self, *args, **kwargs):
        # TODO: separate logger class

        if not self.config.use_wandb:
            return

        if wandb.run is None:
            warn(
                "Weights and biases config was provided, but wandb.init was not called."
            )
            return

        wandb.log(*args, **kwargs)
