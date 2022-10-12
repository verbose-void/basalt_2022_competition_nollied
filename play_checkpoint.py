from fgz.training.fgz_trainer import FGZTrainer
from train import get_agent

agent = get_agent()
trainer = FGZTrainer.load("fgz_dynamics_checkpoint.pth", agent)

trainer.evaluate("MineRLBasaltMakeWaterfall-v0", render=True, max_steps=256, force_no_escape=True)