import ray

from fgz.training.fgz_trainer import FGZTrainer
from fgz_config import TASKS

import torch

@ray.remote
class Evaluator:

    def evaluate(self, path_to_checkpoint) -> str:
        trainer = FGZTrainer.load(path_to_checkpoint, device=torch.device("cpu"))

        task_id = trainer.config.enabled_tasks[0]
        eval_env_id = TASKS[task_id]["dataset_dir"]

        video_filepath = trainer.evaluate(eval_env_id, render=False, save_video=True, max_steps=32, force_no_escape=True)
        return video_filepath
