import numpy as np

from fractal_zero.config import FractalZeroConfig


class GameHistory:
    def __init__(self, initial_observation):
        # first "frame" is always empty (with the actual initial observation)
        self.actions = [0]  # TODO: use the action shape
        self.observations = [initial_observation]
        self.environment_reward_signals = [0]
        self.values = [0]

    @property
    def observation_shape(self):
        return self.observations[0].shape

    @property
    def total_reward(self):
        return sum(self.environment_reward_signals)

    @property
    def action_shape(self):
        return tuple()  # TODO!

    def append(self, action, observation, environment_reward_signal, value):
        self.actions.append(action)
        self.observations.append(observation)
        self.environment_reward_signals.append(environment_reward_signal)
        self.values.append(value)

    def __getitem__(self, index: int):
        return (
            self.observations[index],
            self.actions[index],
            self.environment_reward_signals[index],
            self.values[index],
        )

    def __len__(self):
        if (
            len(self.observations)
            == len(self.actions)
            == len(self.environment_reward_signals)
        ):
            return len(self.observations)
        raise ValueError(str(self))

    def __str__(self):
        return f"GameHistory(num_actions={len(self.actions)}, num_observation={len(self.observations)}, num_rewards={len(self.environment_reward_signals)})"


class ReplayBuffer:
    def __init__(self, config: FractalZeroConfig):
        # TODO: prioritized experience replay (PER) https://arxiv.org/abs/1511.05952

        self.config = config

        self.game_histories = []

    def append(self, game_history: GameHistory):
        """Add a trajectory/episode to the replay buffer. If the buffer is full, a trajectory will be popped according
        to the pop strategy specified in the config.
        """

        if len(self) >= self.config.max_replay_buffer_size:
            strat = self.config.replay_buffer_pop_strategy

            if strat == "oldest":
                i = 0
            elif strat == "random":
                i = np.random.randint(0, len(self))
            else:
                raise NotImplementedError(
                    f'Replay buffer pop strategy "{strat}" is not supported.'
                )

            self.game_histories.pop(i)

        self.game_histories.append(game_history)

        if len(self) > self.config.max_replay_buffer_size:
            raise ValueError

    def sample_game(self) -> GameHistory:
        game_index = np.random.randint(0, len(self.game_histories))
        return self.game_histories[game_index]

    def sample_game_clip(
        self, clip_length: int, pad_to_num_frames: bool = True
    ) -> tuple:
        assert clip_length > 0

        game = self.sample_game()

        # minimizing padding means the start frame chosen will result in the least amount of padded frames.
        if self.config.minimize_batch_padding:
            if len(game) <= clip_length:
                start_frame = 0
            else:
                start_frame = np.random.randint(0, len(game) - clip_length)
        else:
            start_frame = np.random.randint(0, len(game))

        end_frame = start_frame + clip_length

        actual_frames = game[start_frame:end_frame]

        actual_num_frames = len(actual_frames[0])
        num_frames = clip_length if pad_to_num_frames else actual_num_frames

        observations = np.zeros((num_frames, *game.observation_shape), dtype=float)
        actions = np.zeros(
            (
                num_frames,
                *game.action_shape,
            ),
            dtype=float,
        )
        rewards = np.zeros((num_frames,), dtype=float)
        values = np.zeros((num_frames,), dtype=float)

        observations[:actual_num_frames] = actual_frames[0]
        actions[:actual_num_frames] = actual_frames[1]
        rewards[:actual_num_frames] = actual_frames[2]
        values[:actual_num_frames] = actual_frames[3]

        num_empty_frames = clip_length - actual_num_frames

        return observations, actions, rewards, values, num_empty_frames

    def get_episode_lengths(self):
        return [len(history) for history in self.game_histories]

    def __len__(self):
        return len(self.game_histories)
