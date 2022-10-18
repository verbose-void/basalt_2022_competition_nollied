from datetime import datetime
import torch

import wandb
import os

from fractal_zero.data.data_handler import DataHandler
from fractal_zero.fractal_zero import FractalZero
from fractal_zero.utils import mean_min_max_dict


class FractalZeroTrainer:
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None

    def __init__(
        self,
        fractal_zero: FractalZero,
        data_handler: DataHandler,
    ):
        self.config = fractal_zero.config
        self.data_handler = data_handler
        self.fractal_zero = fractal_zero

        self._setup_optimizer()
        self._setup_lr_schedule()
        self._setup_logger()

        self.completed_train_steps = 0

    def _setup_optimizer(self):
        op = self.config.optimizer.lower()

        if op == "sgd":
            self.optimizer = torch.optim.SGD(
                self.fractal_zero.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
            )
        elif op == "adam":
            self.optimizer = torch.optim.Adam(
                self.fractal_zero.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(f'Optimizer "{op}" not yet supported.')

    def _setup_lr_schedule(self):
        lr_config = self.config.lr_scheduler_config
        lr_config.pop("alias")
        scheduler_class = lr_config.pop("class")
        self.lr_scheduler = scheduler_class(self.optimizer, **lr_config)

    def _setup_logger(self):
        if self.config.use_wandb:
            if "config" in self.config.wandb_config:
                raise KeyError(
                    "The config field in `wandb_config` will be automatically set using the FractalZeroConfig."
                )
            wandb.init(**self.config.wandb_config, config=self.config.asdict())
            self.run_name = wandb.run.name
        else:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    @property
    def representation_model(self):
        return self.fractal_zero.model.representation_model

    @property
    def dynamics_model(self):
        return self.fractal_zero.model.dynamics_model

    @property
    def prediction_model(self):
        return self.fractal_zero.model.prediction_model

    def _get_batch(self):
        batch = self.data_handler.get_batch(self.config.unroll_steps)

        (
            self.observations,
            self.actions,
            self.target_auxiliaries,
            self.target_values,
            self.num_empty_frames,
        ) = batch

        return batch

    def _unroll(self):
        # TODO: docstring

        first_observations = self.observations[:, 0]
        first_hidden_states = self.representation_model.forward(first_observations)

        self.dynamics_model.set_state(first_hidden_states)

        # preallocate unroll prediction arrays
        self.unrolled_auxiliaries = torch.zeros_like(self.target_auxiliaries)
        self.unrolled_values = torch.zeros_like(self.target_values)

        # fill arrays
        for unroll_step in range(self.config.unroll_steps):
            step_actions = self.actions[:, unroll_step]
            self.unrolled_auxiliaries[:, unroll_step] = self.dynamics_model(
                step_actions
            )

            state = self.dynamics_model.state
            _, value_predictions = self.prediction_model.forward(state)
            # TODO: unroll policy

            self.unrolled_values[:, unroll_step] = value_predictions

    def _calculate_losses(self):
        auxiliary_loss = (
            self.dynamics_model.auxiliary_loss(
                self.unrolled_auxiliaries,
                self.target_auxiliaries,
            )
            / self.config.unroll_steps
        )

        value_loss = (
            self.prediction_model.value_loss(self.unrolled_values, self.target_values)
            / self.config.unroll_steps
        )

        composite_loss = auxiliary_loss + value_loss

        self.log(
            {
                "losses/auxiliary": auxiliary_loss.item(),
                "losses/value": value_loss.item(),
                "losses/composite": composite_loss.item(),
                **mean_min_max_dict("data/auxiliary_targets", self.target_auxiliaries),
                **mean_min_max_dict("data/target_values", self.target_values),
                "data/replay_buffer_size": len(self.data_handler.replay_buffer),
                **mean_min_max_dict(
                    "data/replay_buffer_episode_lengths",
                    self.data_handler.replay_buffer.get_episode_lengths(),
                ),
                "data/batch_size": len(self.target_auxiliaries),
                "data/empty_frames_in_batch": self.num_empty_frames,
                **mean_min_max_dict(
                    "lr/learning_rates", self.lr_scheduler.get_last_lr()
                ),
            }
        )

        return composite_loss

    def train_step(self):
        self.fractal_zero.train()

        self.optimizer.zero_grad()

        self._get_batch()
        self._unroll()

        composite_loss = self._calculate_losses()
        composite_loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.completed_train_steps += 1

    @property
    def checkpoint_filename(self) -> str:
        return f"{self.run_name}.checkpoint"

    def save_checkpoint(self, folder: str = "checkpoints") -> str:
        # TODO: optionally save to wandb

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.checkpoint_filename)
        torch.save(self, path)
        return path

    def log(self, *args, **kwargs):
        if self.config.use_wandb:
            wandb.log(*args, **kwargs)
