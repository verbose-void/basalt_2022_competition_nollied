from warnings import warn
import torch

from fgz.architecture.xirl_model import XIRLModel
from fgz.data_utils.data_handler import ChunkedContiguousTrajectory
from xirl_config import XIRLConfig
from tqdm import tqdm


class TrajectoryBatcher:
    def __init__(self, trajectory: ChunkedContiguousTrajectory, batch_size: int):
        self.trajectory = trajectory
        self.batch_size = batch_size

        self.num_frames = len(trajectory)

    def __iter__(self):
        self._batch = 0
        self._iterator = iter(self.trajectory)
        return self

    def __next__(self):
        # accumulate enough samples for the current batch
        batch = []

        for _ in range(self.batch_size):
            try:
                frame, action = self._iterator.__next__()
                batch.append(frame)

            except StopIteration:
                break

        return torch.tensor(batch)

    def __len__(self):
        return self.num_frames // self.batch_size



@torch.no_grad()
def plot_xirl_reward_over_time(config: XIRLConfig, trajectory: ChunkedContiguousTrajectory, model: XIRLModel, target_embedding: torch.Tensor):

    # need to first gather the target embedding from all of the videos
    # then need to load a video from the dataset and embed all of the frames
    # then i need to calculate the reward for each frame, which is the negative distance between the embeddings

    # i guess i can just start off by plotting the reward over time.

    all_distances = []

    batch_size = 32 # config.embed_batch_size
    batcher = TrajectoryBatcher(trajectory, batch_size)
    for frame_batch in tqdm(batcher, desc="Getting rewards", total=len(batcher)):
        if frame_batch.numel() <= 0:
            warn("Empty batch....")
            continue

        # print(frame_batch.shape)
        embedded = model.embed(frame_batch)
        target_embedding.expand(len(embedded), -1)

        distances = torch.norm(embedded - target_embedding, dim=1)
        # print(distances.shape, distances.mean())
        all_distances.append(distances)

    all_distances = torch.concat(all_distances)
    print(all_distances)
    return all_distances

