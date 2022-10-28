

from xirl_zero.main_trainer import Config, Trainer


if __name__ == "__main__":
    dataset_path = "/Volumes/CORSAIR/data/MineRLBasaltMakeWaterfall-v0"  # TODO: determine based on task idx

    config = Config(dataset_path=dataset_path)
    trainer = Trainer(config)
    trainer.train_step()