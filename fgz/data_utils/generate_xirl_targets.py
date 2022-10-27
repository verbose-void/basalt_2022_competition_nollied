from warnings import warn
from fgz.architecture.xirl_model import XIRLModel
from fgz.data_utils.data_handler import ContiguousTrajectoryDataLoader
from xirl_config import XIRLConfig

from tqdm import tqdm

import torch



@torch.no_grad()
def generate_target(config: XIRLConfig, model: XIRLModel, dataset_path: str):
    trajectory_loader = ContiguousTrajectoryDataLoader(dataset_path)

    bs = config.embed_batch_size
    target_embedding = torch.zeros(2048, dtype=float)

    all_last_frames = []

    def _process_batch():
        batch = torch.tensor(all_last_frames, device=torch.device("cuda"))
        embedded_batch = model.embed(batch)
        all_last_frames.clear()
        return torch.sum(embedded_batch, dim=0).cpu()

    num_demonstrations = 0
    for trajectory in tqdm(trajectory_loader.trajectories, desc="Loading last frames", total=len(trajectory_loader.trajectories)):
        try:
            all_last_frames.append(trajectory.get_last_frame())
            num_demonstrations += 1
        except:
            warn(f"Skipping {str(trajectory)}")
            pass

        if len(all_last_frames) >= bs:
            target_embedding += _process_batch()
    if len(all_last_frames) > 0:
        target_embedding += _process_batch()

    target_embedding /= num_demonstrations

    # all_last_frames = torch.tensor(all_last_frames)


    # target_embedding = torch.zeros(2048, dtype=float)
    # num_demonstrations = len(all_last_frames)

    # c = 0
    # while c < len(all_last_frames):
    #     print(f"{c / len(all_last_frames) * 100}\%")

    #     batch = all_last_frames[c:c+bs]

    #     embedded_batch = model.embed(batch)
    #     embedded_batch /= num_demonstrations
    #     target_embedding += torch.sum(embedded_batch, dim=0)

    #     c += bs

    # # model.embed(all_last_frames)
    return target_embedding
