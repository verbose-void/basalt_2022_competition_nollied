from time import sleep

import torch
from fractal_zero.config import FractalZeroConfig

from fractal_zero.data.replay_buffer import GameHistory
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)


class FractalZero(torch.nn.Module):
    def __init__(self, config: FractalZeroConfig):
        super().__init__()

        self.config = config

        self.model = self.config.joint_model

        # TODO: reuse FMC instance?
        self.fmc = None

        self.actual_env = self.config.env

        # TODO: explain
        if self.fmc_config.search_using_actual_environment:
            self.vectorized_environment = RayVectorizedEnvironment(
                self.actual_env,
                n=self.fmc_config.num_walkers,
            )
        else:
            self.vectorized_environment = VectorizedDynamicsModelEnvironment(
                self.actual_env,
                n=self.fmc_config.num_walkers,
                joint_model=self.model,
            )

    @property
    def fmc_config(self):
        return self.config.fmc_config

    def forward(self, observation):
        # TODO: docstring, note that lookahead_steps == 0 means there won't be a tree search

        if self.training:
            greedy_action = False
            k = self.config.lookahead_steps
        else:
            greedy_action = True
            k = self.config.evaluation_lookahead_steps

        if self.config.lookahead_steps > 0:
            self.vectorized_environment.set_all_states(self.actual_env, observation)
            action = self.fmc.simulate(k, greedy_action=greedy_action)
            return action, self.fmc.root_value

        raise NotImplementedError("Action prediction not yet working.")

    def play_game(
        self,
        render: bool = False,
    ):
        obs = self.actual_env.reset()
        game_history = GameHistory(obs)

        self.fmc = FMC(
            self.vectorized_environment,
            self.model.prediction_model,
            config=self.fmc_config,
        )

        for step in range(self.config.max_game_steps):
            obs = torch.tensor(obs, device=self.config.device)
            action, root_value = self.forward(obs)
            obs, reward, done, info = self.actual_env.step(action)

            game_history.append(action, obs, reward, root_value)

            if render:
                print()
                print(f"step={step}")
                print(f"reward={reward}, done={done}, info={info}")
                print(
                    f"action={action}, root_value={root_value}"  # , value_estimate={value_estimate}"
                )
                self.actual_env.render()
                sleep(0.1)

            if done:
                break

        if render:
            print()
            print("game summary:")
            print(f"cumulative rewards: {sum(game_history.environment_reward_signals)}")
            print(f"episode length: {len(game_history)}")

        return game_history
