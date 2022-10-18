import gym

from fractal_zero.config import FMCConfig, FractalZeroConfig
from fractal_zero.data.data_handler import DataHandler
from fractal_zero.fractal_zero import FractalZero

from fractal_zero.models.dynamics import FullyConnectedDynamicsModel
from fractal_zero.models.joint_model import JointModel
from fractal_zero.models.prediction import FullyConnectedPredictionModel
from fractal_zero.models.representation import FullyConnectedRepresentationModel
from fractal_zero.trainer import FractalZeroTrainer

import wandb
from tqdm import tqdm

from fractal_zero.utils import mean_min_max_dict


def get_cartpole_joint_model(env: gym.Env, embedding_size: int = 16) -> JointModel:
    out_features = 1

    representation_model = FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(
        env, embedding_size, out_features=out_features
    )
    prediction_model = FullyConnectedPredictionModel(env, embedding_size)
    return JointModel(representation_model, dynamics_model, prediction_model)


def get_cartpole_config(
    env: gym.Env, alphazero_style: bool, use_wandb: bool
) -> FractalZeroConfig:
    if alphazero_style:
        joint_model = get_cartpole_joint_model(env, embedding_size=4)
    else:
        joint_model = get_cartpole_joint_model(env, embedding_size=16)

    if use_wandb:
        wandb_config = {"project": "fractal_zero_cartpole"}
    else:
        wandb_config = None

    fmc_config = FMCConfig(
        num_walkers=16,
        balance=1.0,
        search_using_actual_environment=alphazero_style,
        use_wandb=wandb_config is not None,
    )

    return FractalZeroConfig(
        env,
        joint_model,
        fmc_config=fmc_config,
        max_replay_buffer_size=64,
        num_games=1_024,
        max_game_steps=200,
        max_batch_size=16,
        unroll_steps=8,
        learning_rate=0.003,
        optimizer="SGD",
        weight_decay=1e-4,
        momentum=0.9,  # only if optimizer is SGD
        lookahead_steps=8,
        evaluation_lookahead_steps=8,
        wandb_config=wandb_config,
    )


def train_cartpole(alphazero_style: bool, use_wandb: bool):
    env = gym.make("CartPole-v0")
    config = get_cartpole_config(env, alphazero_style, use_wandb)

    # TODO: move into config?
    train_every = 1
    train_batches = 2
    evaluate_every = 16
    eval_steps = 16

    # TODO: make this logic automatic in config somehow?
    config.joint_model = config.joint_model.to(config.device)

    data_handler = DataHandler(config)
    fractal_zero = FractalZero(config)
    trainer = FractalZeroTrainer(
        fractal_zero,
        data_handler,
    )

    for i in tqdm(
        range(config.num_games),
        desc="Playing games and training",
        total=config.num_games,
    ):
        fractal_zero.train()
        game_history = fractal_zero.play_game()
        data_handler.replay_buffer.append(game_history)

        if i % train_every == 0:
            for _ in range(train_batches):
                trainer.train_step()

        if i % evaluate_every == 0:
            # TODO: move into trainer?

            fractal_zero.eval()

            game_lengths = []
            cumulative_rewards = []
            for _ in range(eval_steps):
                game_history = fractal_zero.play_game()
                game_lengths.append(len(game_history))
                cumulative_reward = sum(game_history.environment_reward_signals)
                cumulative_rewards.append(cumulative_reward)

            if config.use_wandb:
                wandb.log(
                    {
                        **mean_min_max_dict("evaluation/episode_length", game_lengths),
                        **mean_min_max_dict(
                            "evaluation/cumulative_reward", cumulative_rewards
                        ),
                    },
                    commit=False,
                )

            # checkpoint after each evaluation
            trainer.save_checkpoint()


if __name__ == "__main__":
    alphazero_style = False
    use_wandb = False
    train_cartpole(alphazero_style, use_wandb)
