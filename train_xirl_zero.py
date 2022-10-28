

import wandb
from xirl_zero.main_trainer import Config, Trainer


SMOKE_TEST = False


if __name__ == "__main__":
    # TODO: determine based on task idx
    # dataset_path = "/Volumes/CORSAIR/data/MineRLBasaltMakeWaterfall-v0"
    dataset_path = "./data/MineRLBasaltMakeWaterfall-v0"

    config = Config(
        dataset_path=dataset_path,
        max_frames=10 if SMOKE_TEST else None,
        use_wanbd=True,
    )

    trainer = Trainer(config)

    if config.use_wanbd:
        wandb.init(project="xirl_zero", config={
            "main_config": config.__dict__,
            "representation_config": trainer.representation_trainer.config.__dict__,
            "dynamics_config": trainer.dynamics_trainer.config.__dict__,
        })

    def run_eval(steps: int):
        for _ in range(steps):
            trainer.eval_step()

    def run_train(steps: int, eval_every: int, eval_steps: int):
        for step in range(steps):
            trainer.train_step()

            if step % eval_every == 0:
                run_eval(eval_steps)

                # TODO: calculate target state and visualize in wandb.
                # TODO: save models with the best evaluation performance.
                # TODO: save the target embedding for the model with the best evaluation performance.

    train_steps = 10_000
    eval_every = 100
    eval_steps = 10
    run_train(steps=train_steps, eval_every=eval_every, eval_steps=eval_steps)

    train_target_state = trainer.get_target_state()
    print(train_target_state)
