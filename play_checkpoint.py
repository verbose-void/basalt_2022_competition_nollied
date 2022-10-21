from fgz.training.fgz_trainer import FGZTrainer
from train import get_agent

trainer = FGZTrainer.load("train/fgz_dynamics_checkpoint.pth")

trainer.evaluate(
    "MineRLBasaltMakeWaterfall-v0", render=True, max_steps=256, force_no_escape=True
)
