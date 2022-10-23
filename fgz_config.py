import minerl
from dataclasses import dataclass, field
import os
from typing import List
import gym


ACTION_SPACE = gym.make("MineRLBasaltMakeWaterfall-v0").action_space


TASKS = [
    {"name": "BuildHouse", "dataset_dir": "MineRLBasaltBuildVillageHouse-v0"},
    {"name": "AnimalPen", "dataset_dir": "MineRLBasaltCreateVillageAnimalPen-v0"},
    {"name": "FindCave", "dataset_dir": "MineRLBasaltFindCave-v0"},
    {"name": "BuildWaterfall", "dataset_dir": "MineRLBasaltMakeWaterfall-v0"},
]


MINERL_DATA_ROOT = os.getenv("MINERL_DATA_ROOT", "data/")
VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")


@dataclass
class FGZConfig:
    # model_filename: str = "foundation-model-1x.model"
    model_filename: str = "foundation-model-2x.model"
    # model_filename: str = "foundation-model-3x.model"

    # weights_filename: str = "foundation-model-1x.weights"
    weights_filename: str = "rl-from-early-game-2x.weights"

    # enable tasks according to the indices of the `TASKS` list. 0="BuildHouse", etc.
    # ordering of these tasks determines the ordering of the responsibility of the discriminator logits.
    # ENABLED_TASKS = [0, 1, 2, 3]  # all 4 tasks.

    enabled_tasks: List[int] = field(default_factory=lambda: [3])  # waterfall only.

    # when this is true, FMC will not be run and the discriminator will only try to discriminate between
    # the enabled tasks.
    disable_fmc_detection: bool = False

    verbose: bool = False

    batch_size: int = 64
    learning_rate = 0.00008
    consistency_loss_coeff: float = 0.0  # if 0, consistency loss is ignored.

    num_walkers: int = 128
    fmc_steps: int = 16
    unroll_steps: int = 4
    fmc_random_policy: bool = False

    use_wandb: bool = False

    action_space = ACTION_SPACE

    @property
    def fmc_logit(self):
        # the last logit is dedicated to FMC
        return len(self.enabled_tasks)

    @property
    def num_discriminator_classes(self):
        # plus 1 because FMC is a class label
        if self.disable_fmc_detection:
            return len(self.enabled_tasks)
        return len(self.enabled_tasks) + 1

    @property
    def model_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.model_filename)

    @property
    def weights_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.weights_filename)

    @property
    def dataset_paths(self) -> List[str]:
        return [
            os.path.join(MINERL_DATA_ROOT, TASKS[task_id]["dataset_dir"])
            for task_id in self.enabled_tasks
        ]

    @property
    def environment_id_to_task_logit(self):
        return {
            TASKS[task_id]["dataset_dir"]: i
            for i, task_id in enumerate(self.enabled_tasks)
        }

    def asdict(self):
        return {**self.__dict__, "enabled_tasks": self.enabled_tasks}


if __name__ == "__main__":
    config = FGZConfig().asdict()
    print(config)
