import torch
import numpy as np

from fractal_zero.search.tree import GameTree, StateNode

import wandb


class TreeSampler:
    def __init__(
        self,
        tree: GameTree,
        sample_type: str = "all_nodes",
        weight_type: str = "walker_children_ratio",
        use_wandb: bool = False,
    ):
        self.tree = tree
        self.sample_type = sample_type
        self.weight_type = weight_type
        self.use_wandb = use_wandb

        if not self.tree.prune:
            raise NotImplementedError(
                "TreeSampling on an unpruned tree has not been considered."
            )

    def _calculate_weight(self, node: StateNode) -> float:
        if self.weight_type == "walker_children_ratio":
            return node.num_child_walkers / self.tree.num_walkers
        elif self.weight_type == "constant":
            return 1.0
        elif self.weight_type == "time_spent_at_node":
            raise NotImplementedError
        raise ValueError(f"{self.weight_type} not supported.")

    def _get_best_path_as_batch(self):
        observations = []
        actions = []
        weights = []
        rewards = []
        infos = []

        path = self.tree.best_path
        for state, action in path:

            observations.append(state.observation)
            rewards.append(state.reward)

            actions.append([action])

            # NOTE: never skip due to weight!
            weight = self._calculate_weight(state)
            weights.append([weight])
            infos.append(state.info)

        return observations, actions, weights, rewards, infos

    def _get_all_nodes_batch(self):
        observations = []
        child_actions = []
        child_weights = []
        rewards = []
        infos = []

        g = self.tree.g
        for node in g.nodes:

            actions = []
            weights = []
            for _, child_node, data in g.out_edges(node, data=True):
                weight = self._calculate_weight(child_node)

                # skip if the weight is almost 0.
                if np.isclose(weight, 0):
                    continue

                weights.append(weight)
                action = data["action"]
                actions.append(action)

            # if no action targets exist, skip this state.
            if len(actions) <= 0:
                continue

            observations.append(node.observation)
            child_actions.append(actions)
            child_weights.append(weights)
            rewards.append(node.reward)
            infos.append(node.info)

        return observations, child_actions, child_weights, rewards, infos

    def get_batch(self):
        if self.sample_type == "best_path":
            obs, acts, weights, rewards, infos = self._get_best_path_as_batch()
        elif self.sample_type == "all_nodes":
            obs, acts, weights, rewards, infos = self._get_all_nodes_batch()
        else:
            raise ValueError(f"Sample type {self.sample_type} is not supported.")

        # sanity check
        if not (len(obs) == len(acts) == len(weights) == len(rewards)):
            raise ValueError(
                f"Got different lengths for batch return: {len(obs)}, {len(acts)}, {len(weights)}, {len(rewards)}."
            )

        if self.use_wandb and wandb.run:
            mean_weight = np.mean([np.mean(w) for w in weights])
            mean_num_actions = np.mean([len(act) for act in acts])

            wandb.log(
                {
                    "tree_sampler/mean_weights": mean_weight,
                    "tree_sampler/num_samples": len(obs),
                    "tree_sampler/mean_num_actions": mean_num_actions,
                }
            )

        return obs, acts, weights, rewards, infos
