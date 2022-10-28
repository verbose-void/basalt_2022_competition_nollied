import torch

from xirl_zero.main_trainer import Trainer

import os



class Tester:

    def __init__(self, path_to_experiment: str, iteration: int=None, device=None):
        checkpoint_dir = os.path.join(path_to_experiment, "checkpoints")

        # default pick the last iteration
        if iteration is None:
            iterations = [int(fn.split(".")[0]) for fn in os.listdir(checkpoint_dir)]
            iteration = max(iterations)
        
        fn = f"{iteration}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, fn)
        target_state_path = os.path.join(path_to_experiment, "target_states", fn)

        print(f"Loading {fn} checkpoint and target state from {path_to_experiment}")

        trainer: Trainer = torch.load(checkpoint_path, map_location=device)
        target_state: torch.Tensor = torch.load(target_state_path, map_location=device)

        self.representation_model = trainer.representation_trainer.model
        self.dynamics_model = trainer.dynamics_trainer.model
        self.target_state = target_state


if __name__ == "__main__":
    tester = Tester("./train/xirl_zero/2022-10-28_02-14-43_PM/")
