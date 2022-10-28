

from xirl_zero.main_trainer import Config, Trainer


if __name__ == "__main__":
    # TODO: determine based on task idx
    # dataset_path = "/Volumes/CORSAIR/data/MineRLBasaltMakeWaterfall-v0"
    dataset_path = "./data/MineRLBasaltMakeWaterfall-v0"

    config = Config(dataset_path=dataset_path)
    trainer = Trainer(config)

    def run_train(steps: int):
        for _ in range(steps):
            trainer.train_step()

    def run_eval(steps: int):
        for _ in range(steps):
            trainer.eval_step()


    train_steps = 10
    run_train(train_steps)
    
    eval_steps = 10
    run_eval(eval_steps)

    train_target_state = trainer.get_target_state()
    print(train_target_state)