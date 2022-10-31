


from collections import defaultdict
from glob import glob
from logging import warn
from typing import Sequence, Tuple
import os
import torch

import numpy as np

from fgz.data_utils.data_handler import ChunkedContiguousTrajectory, get_trajectories

def read_frames_and_actions(trajectory: ChunkedContiguousTrajectory, num_frame_samples: int, max_frames: int = None):
    # reset hidden state.
    # self.agent.reset()

    frames = []

    # actions are stored as sub-lists of actions, where each sublist corresponds to all actions
    # between the frame samples.
    sup_actions: Sequence[Tuple] = []
    sub_actions = []

    with torch.no_grad():
        # NOTE: some frames may get corrupt during the loading process, 
        # if that occurs not all frames will be used probably. it may not
        # have a huge impact on training, but it's definitely good to be
        # aware of.

        # we can't load more frames than are available
        num_frames = len(trajectory) if max_frames is None else min(max_frames, len(trajectory))
        nframes_to_use = min(num_frame_samples, num_frames)
        frames_to_use = torch.round(torch.linspace(start=0, end=num_frames, steps=nframes_to_use)).int().tolist()

        c = 0

        last_frame = None
        for i, (frame, action) in enumerate(trajectory):
            if max_frames is not None and i >= max_frames:
                break

            if i in frames_to_use:
                # the first frame will always be included, but never have populated sub_actions.
                if i > 0:
                    sup_actions.append(tuple(sub_actions))

                sub_actions = [action]

            if i not in frames_to_use:
                # NOTE: this is a hack -- it would be faster to not load these frames from the disk at all.
                sub_actions.append(action)
                continue

            frame_tensor = torch.tensor(frame)
            frames.append(frame_tensor.float())

            c += 1

            last_frame = frame_tensor

        # print(frames_to_use)
        # print(len(frames_to_use))
        assert len(frames) <= len(frames_to_use)

        # ensure we always have the last frame in the sequence.
        if last_frame is not None and c < len(frames_to_use):
            frames.append(last_frame)

    if len(sub_actions) > 0:
        sup_actions.append(tuple(sub_actions))

    frames = torch.stack(frames)
    frames.requires_grad = True

    return frames, sup_actions


def get_trajectories(dataset_path: str, train_split: float):
    assert train_split > 0 and train_split < 1

    # gather all unique IDs for every video/json file pair.
    unique_ids = glob(os.path.join(dataset_path, "*.mp4"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))

    # get all chunked/unchunked trajectories as a dict of lists
    full_trajectory_ids = defaultdict(list)
    for clip_uid in unique_ids:

        # player_uid, game_uid, date, time = clip_uid.split("-")  # doesn't work for "cheeky-cornflower" stuff
        splitted = clip_uid.split("-")
        game_uid, date, time = splitted[-3:]
        player_uid = "-".join(splitted[:-3])

        trajectory_prefix = os.path.join(dataset_path, f"{player_uid}-{game_uid}")
        full_trajectory_ids[trajectory_prefix].append((date, time))

    max_train_idx = int(np.floor(len(full_trajectory_ids) * train_split))

    # training gets the first section, eval gets last
    train_trajectories: Sequence[ChunkedContiguousTrajectory] = []
    eval_trajectories: Sequence[ChunkedContiguousTrajectory] = []

    for i, trajectory_prefix in enumerate(sorted(full_trajectory_ids.keys())):
        date_times = full_trajectory_ids[trajectory_prefix]

        sorted_date_times = list(sorted(date_times))
        # print(trajectory_prefix, sorted_date_times)

        traj_list = train_trajectories if i <= max_train_idx else eval_trajectories

        try:
            chunked_traj = ChunkedContiguousTrajectory(trajectory_prefix, sorted_date_times, trajectory_prefix)#, task_id=task_id)
            traj_list.append(chunked_traj)
        except ValueError:
            warn(f"Missing video/json path! Skipping... {trajectory_prefix} {sorted_date_times}")
            continue

    return train_trajectories, eval_trajectories


class ContiguousTrajectoryLoader:
    trajectories: Sequence[ChunkedContiguousTrajectory]

    def __init__(self, trajectories: Sequence[ChunkedContiguousTrajectory]):
        self.trajectories = trajectories

        self.minimum_steps = 64

    def sample_trajectory_object(self) -> ChunkedContiguousTrajectory:
        trajectory_index = np.random.randint(low=0, high=len(self.trajectories))
        t = self.trajectories[trajectory_index]

        # remove the trajectory and pick a new sample if it's not long enough.
        t_len = len(t)
        if t_len < self.minimum_steps:
            self.trajectories.pop(trajectory_index)
            warn(
                f"Removing trajectory from the dataset. It's length was too short ({t_len} < {self.minimum_steps}). Now there are {len(self.trajectories)} left."
            )
            return self.sample_trajectory_object()
        return t

    def sample(self, num_frame_samples: int, max_frames: int=None):
        t = self.sample_trajectory_object()
        return read_frames_and_actions(t, num_frame_samples=num_frame_samples, max_frames=max_frames)

    @staticmethod
    def get_train_and_eval_loaders(dataset_path: str, train_split: float=0.8, max_trajectories: int = None):
        train_trajectories, eval_trajectories = get_trajectories(dataset_path, train_split=train_split)

        if max_trajectories is not None:
            train_trajectories = train_trajectories[:max_trajectories]
            eval_trajectories = eval_trajectories[:max_trajectories]

        train_loader = ContiguousTrajectoryLoader(train_trajectories)
        eval_loader = ContiguousTrajectoryLoader(eval_trajectories)

        return train_loader, eval_loader