

import wandb
from xirl_zero.main_trainer import Config, Trainer

import torch


SMOKE_TEST = True


if __name__ == "__main__":
    # TODO: determine based on task idx
    # dataset_path = "/Volumes/CORSAIR/data/MineRLBasaltMakeWaterfall-v0"
    dataset_path = "./data/MineRLBasaltMakeWaterfall-v0"

    output_dir = "./train/xirl_zero/"

    config = Config(
        dataset_path=dataset_path,
        max_frames=10 if SMOKE_TEST else None,
        max_trajectories=10 if SMOKE_TEST else None,
        use_wandb=False,
        model_log_frequency=1 if SMOKE_TEST else 1000,
    )

    if config.use_wandb:
        wandb.init(project="xirl_zero", config=config.asdict())

    trainer = Trainer(config)

    def run_eval(steps: int):
        for _ in range(steps):
            trainer.eval_step()

    def run_train(steps: int, eval_every: int, eval_steps: int):
        for step in range(steps):
            trainer.train_step()

            if (step + 1) % eval_every == 0 or step == (steps - 1):
                run_eval(eval_steps)
                trainer.checkpoint(output_dir)
                _, target_state = trainer.generate_and_save_target_state(output_dir)
                print(target_state)

    train_steps = 5 if SMOKE_TEST else 10_000
    eval_every = 1 if SMOKE_TEST else 100
    eval_steps = 5 if SMOKE_TEST else 100
    run_train(steps=train_steps, eval_every=eval_every, eval_steps=eval_steps)

    # train_target_state = trainer.get_target_state()
    # print(train_target_state)
