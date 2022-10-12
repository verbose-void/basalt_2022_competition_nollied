import os

MODEL_FILE = "foundation-model-1x.model"
# MODEL_FILE = "foundation-model-2x.model"
# MODEL_FILE = "foundation-model-3x.model"

WEIGHTS_FILE = "foundation-model-1x.weights"
# WEIGHTS_FILE = "rl-from-early-game-2x.weights"

TASKS = [
    {"name": "BuildHouse", "dataset_dir": "MineRLBasaltBuildVillageHouse-v0"},
    {"name": "AnimalPen", "dataset_dir": "MineRLBasaltCreateVillageAnimalPen-v0"},
    {"name": "FindCave", "dataset_dir": "MineRLBasaltFindCave-v0"},
    {"name": "BuildWaterfall", "dataset_dir": "MineRLBasaltMakeWaterfall-v0"},
]

# enable tasks according to the indices of the `TASKS` list. 0="BuildHouse", etc.
# ordering of these tasks determines the ordering of the responsibility of the discriminator logits.
# ENABLED_TASKS = [0, 1, 2, 3]  # all 4 tasks.
ENABLED_TASKS = [3]  # waterfall only.

NUM_WALKERS = 32
UNROLL_STEPS = 4

FMC_LOGIT = len(ENABLED_TASKS)  # the last logit is dedicated to FMC
NUM_DISCRIMINATOR_CLASSES = len(ENABLED_TASKS) + 1

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")

DATASET_PATHS = [os.path.join(MINERL_DATA_ROOT, TASKS[task_id]["dataset_dir"]) for task_id in ENABLED_TASKS]

PRETRAINED_AGENT_MODEL_FILE = os.path.join(VPT_MODELS_ROOT, MODEL_FILE)
PRETRAINED_AGENT_WEIGHTS_FILE = os.path.join(VPT_MODELS_ROOT, WEIGHTS_FILE)

MINERL_ENV_TO_TASK_LOGIT = {
    TASKS[task_id]["dataset_dir"]: i for i, task_id in enumerate(ENABLED_TASKS)
}