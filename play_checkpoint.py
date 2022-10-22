import argparse
from fgz.training.fgz_trainer import FGZTrainer
from train import get_agent


parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args().__dict__

trainer = FGZTrainer.load(**args)

# env_id = "MineRLBasaltMakeWaterfall-v0"
env_id = "MineRLBasaltFindCave-v0"

trainer.evaluate(
    env_id, render=True, max_steps=256, force_no_escape=True
)
