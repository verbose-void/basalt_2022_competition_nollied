import numpy as np

from fractal_zero.search.tree import GameTree
import pytest


@pytest.mark.parametrize("prune", [True, False])
def test_tree(prune):
    n = 8
    root_observation = 0
    walker_states = np.ones(n) * root_observation

    def _step(walker_states):
        actions = np.ones(n)
        walker_states += actions
        rewards = np.ones(n)
        return actions, rewards

    tree = GameTree(n, root_observation=root_observation, prune=prune)
    for _ in range(4):
        actions, rewards = _step(walker_states)
        tree.build_next_level(actions, walker_states, rewards)
        assert tree.root.num_child_walkers == n

    # all partners with 0th walker
    partners = np.zeros(n, dtype=int)

    target_state = tree.walker_paths[0].ordered_states[1]
    assert target_state.num_child_walkers == 1

    # only 1 walker clones to the 0th walker
    clone_index = 1
    clone_mask = np.zeros(n, dtype=bool)
    clone_mask[clone_index] = 1

    tree.clone(partners, clone_mask)
    # 1 walker was fully cloned away and pruned.
    assert target_state.num_child_walkers == 2

    if prune:
        assert tree.g.out_degree(tree.root) == n - 1
    else:
        assert tree.g.out_degree(tree.root) == n

    assert tree.g.in_degree(tree.root) == 0
    assert tree.root.num_child_walkers == n
