import gym
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

# from fractal_zero.search.fmc import FMC
from fractal_zero.search.fmc import FMC
from fractal_zero.search.tree import Path
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    SerialVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)

import pytest


USE_TQDM = False


with_vec_envs = pytest.mark.parametrize(
    "vec_env_class", [SerialVectorizedEnvironment, RayVectorizedEnvironment]
)
cloning = pytest.mark.parametrize("disable_cloning", [True, False])

# def _check_last_actions(fmc: FMC):
#     last_actions = fmc.tree.last_actions
#     for last_action, expected_action in zip(last_actions, fmc.actions):
#         assert last_action == expected_action


def _assert_tree_equivalence(fmc: FMC):
    # NOTE: the actions that are in the tree will diverge slightly from
    # those being represented by FMC's internal state. The reason for this is
    # that when we freeze certain environments in the vectorized environment
    # object, the actions that are sampled will not be enacted, and the previous
    # return values will be provided.
    # _check_last_actions(fmc)

    # check rewards are properly matching scores
    # total_rewards = np.array([p.total_reward for p in fmc.tree.walker_paths])
    # expected_total_rewards = fmc.scores.numpy()
    total_rewards = fmc.tree.get_total_rewards()
    expected_total_rewards = fmc.scores

    # np.testing.assert_allclose(total_rewards, expected_total_rewards)
    torch.testing.assert_allclose(total_rewards, expected_total_rewards)

    # TODO: make sure the tree's best path gets the best walker's same score.
    best_walker_index = fmc.scores.argmax()
    score = fmc.scores[best_walker_index]

    expected_best_walker_path = fmc.tree.walker_paths[best_walker_index]
    # assert np.isclose(score, expected_best_walker_path.total_reward)
    assert np.isclose(score.item(), expected_best_walker_path.total_reward)
    assert expected_best_walker_path == fmc.tree.best_path


def _assert_mean_total_rewards(fmc: FMC, steps, expected_mean_reward, trials=32):
    total_rewards = []

    for _ in range(trials):
        fmc.reset()

        # 1 step at a time
        for _ in tqdm(range(steps), disable=not USE_TQDM):
            fmc.simulate(1)
            _assert_tree_equivalence(fmc)
            # cartpole has a max reward of 200.
            assert (fmc.scores <= 200).all()

            if fmc.did_early_exit:
                break

        total_reward = fmc.tree.best_path.total_reward
        total_rewards.append(total_reward)

    assert np.mean(total_rewards) > expected_mean_reward


def _tree_structural_assertions(fmc: FMC, steps: int):
    with_freeze = fmc.freeze_best
    prune = fmc.prune_tree
    n = fmc.num_walkers

    if with_freeze:
        c = n * steps - steps + 1
    else:
        c = n * steps

    if prune:
        # the actual values should be significantly lower for environments with decent reward variance.
        assert fmc.tree.g.number_of_edges() <= c
        assert fmc.tree.g.number_of_nodes() <= c + 1

        # ensure no straggler nodes exist
        for node in fmc.tree.g.nodes:
            assert node.num_child_walkers > 0
            assert node.visits > 0
    else:
        assert fmc.tree.g.number_of_edges() == c
        assert fmc.tree.g.number_of_nodes() == c + 1

    assert fmc.tree.g.in_degree(fmc.tree.root) == 0
    for path in fmc.tree.walker_paths:
        assert path.g == fmc.tree.g
        assert path.ordered_states[0] == path.root == fmc.tree.root
        last_state = None
        for state in path.ordered_states:
            assert state in fmc.tree.g, "Don't over-prune."
            if last_state is not None:
                assert fmc.tree.g.has_edge(
                    last_state, state
                ), "All nodes on the same path must be linearly connected."
                assert path.g.in_degree(state) == 1
            last_state = state

    assert nx.is_tree(fmc.tree.g)


@cloning
@pytest.mark.parametrize("with_freeze", [True, False])
@pytest.mark.parametrize("prune", [False, True])
# @with_vec_envs
def test_cloning(with_freeze, prune, disable_cloning):
    class DummyEnvironment:
        def __init__(self):
            self.reset()
            self.action_space = gym.spaces.Discrete(3)

        def reset(self):
            self.state = 0
            return self.state

        def step(self, action):
            self.state += action
            return float(self.state), action, False, {}

    n = 16
    steps = 16
    # vec_env = vec_env_class(DummyEnvironment(), n=n)
    vec_env = SerialVectorizedEnvironment(DummyEnvironment(), n=n)

    fmc = FMC(
        vec_env,
        freeze_best=with_freeze,
        prune_tree=prune,
        disable_cloning=disable_cloning,
    )

    np.testing.assert_allclose(fmc.scores.numpy(), fmc.tree.get_total_rewards())

    num_clones = 0
    for step in range(steps):
        fmc.simulate(1)
        num_clones += fmc.clone_mask.sum()
        upper_bound_score = (step + 1) * 2

        assert (fmc.scores <= upper_bound_score).all()
        assert len(fmc.similarities) == fmc.num_walkers
        assert len(fmc.scores) == fmc.num_walkers
        assert len(fmc.actions) == fmc.num_walkers

        if disable_cloning and not with_freeze:
            assert fmc.actions == fmc.tree.last_actions

        assert fmc.states.tolist() == fmc.observations
        np.testing.assert_allclose(fmc.scores.numpy(), fmc.states.numpy())
        np.testing.assert_allclose(fmc.scores.numpy(), fmc.tree.get_total_rewards())

        # when no walkers are frozen, the depths should all be consistent.
        d = fmc.tree.get_depths()
        if with_freeze:
            assert torch.all(d <= (step + 1) * 3)
        else:
            np.testing.assert_allclose(d.numpy(), d[0].item())

    # fmc.tree.render()
    _tree_structural_assertions(fmc, steps)

    def _check_tree_observations(path: Path):
        expected_score = 0
        for state, action in path:
            assert state.observation == expected_score
            expected_score += action  # yes, we're checking this because clones have the potential to disrupt in the case of a bug.
        expected_score += path.ordered_states[-1].reward
        assert (
            expected_score == path.total_reward
        ), "Iterating through the path should only yield the states that have following states (with action transitions)."

    # ensure that if you replay the best path, the score is as expected.
    for path in fmc.tree.walker_paths:
        _check_tree_observations(path)


@cloning
@with_vec_envs
def test_cartpole_actual_environment(vec_env_class, disable_cloning):
    env = gym.make("CartPole-v0")

    n = 16
    vec_env = vec_env_class(env, n=n)
    fmc = FMC(vec_env, disable_cloning=disable_cloning)

    if disable_cloning:
        _assert_mean_total_rewards(fmc, 64, 20)
    else:
        _assert_mean_total_rewards(fmc, 64, 50)


# def test_cartpole_dynamics_function():
#     alphazero_style = False

#     env = gym.make("CartPole-v0")
#     joint_model = build_test_joint_model(env, embedding_size=4)

#     config = FMCConfig(
#         num_walkers=NUM_WALKERS,
#         search_using_actual_environment=alphazero_style,
#     )

#     vec_env = VectorizedDynamicsModelEnvironment(env, NUM_WALKERS, joint_model)
#     vec_env.batch_reset()

#     fmc = FMC(vec_env, config=config)
#     fmc.simulate(16)


def test_cartpole_consistently_high_reward():
    n = 64
    vec_env = SerialVectorizedEnvironment("CartPole-v0", n=n)
    fmc = FMC(vec_env, balance=1)

    # 200 is the max reward accumulate-able in cartpole.
    _assert_mean_total_rewards(fmc, 400, 140)
