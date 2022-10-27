from fgz.architecture.xirl_model import XIRLModel
from fgz.data_utils.data_handler import ContiguousTrajectoryDataLoader
from xirl_config import XIRLConfig

from tqdm import tqdm

import torch



def generate_target(config: XIRLConfig, model: XIRLModel, dataset_path: str):
    trajectory_loader = ContiguousTrajectoryDataLoader(dataset_path)

    all_last_frames = []
    for trajectory in trajectory_loader.trajectories:
        all_last_frames.append(trajectory.get_last_frame())
    all_last_frames = torch.tensor(all_last_frames)

    bs = config.embed_batch_size

    target_embedding = torch.zeros(2048, dtype=float)
    num_demonstrations = len(all_last_frames)

    c = 0
    while c < len(all_last_frames):
        batch = all_last_frames[c:c+bs]

        embedded_batch = model.embed(batch)
        embedded_batch /= num_demonstrations
        target_embedding += torch.sum(embedded_batch, dim=0)

        c += bs

    # model.embed(all_last_frames)
    return target_embedding
