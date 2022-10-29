from xirl_zero.main_trainer import Config
from train_xirl_zero import run_train_loop


EVAL_STEPS = 0  # no eval (?)
USE_WANDB = False


WATERFALL_CONFIG = Config(
    minerl_env_id="MineRLBasaltMakeWaterfall-v0",
    train_steps=0,  # TODO
    eval_every=1,
    eval_steps=EVAL_STEPS,
    checkpoint_every=10,  # TODO
    use_wandb=USE_WANDB,
)


CAVE_CONFIG = Config(
    minerl_env_id="MineRLBasaltFindCave-v0",
    train_steps=0,  # TODO
    eval_every=1,
    eval_steps=EVAL_STEPS,
    checkpoint_every=10,  # TODO
    use_wandb=USE_WANDB,
)


ANIMAL_PEN_CONFIG = Config(
    minerl_env_id="MineRLBasaltCreateVillageAnimalPen-v0",
    train_steps=0,  # TODO
    eval_every=1,
    eval_steps=EVAL_STEPS,
    checkpoint_every=10,  # TODO
    use_wandb=USE_WANDB,
)


HOUSE_CONFIG = Config(
    minerl_env_id="MineRLBasaltBuildVillageHouse-v0",
    train_steps=0,  # TODO
    eval_every=1,
    eval_steps=EVAL_STEPS,
    checkpoint_every=10,  # TODO
    use_wandb=USE_WANDB,
)


CONFIGS = [WATERFALL_CONFIG, CAVE_CONFIG, ANIMAL_PEN_CONFIG, HOUSE_CONFIG]


def main():
    for config in CONFIGS:
        run_train_loop(config)