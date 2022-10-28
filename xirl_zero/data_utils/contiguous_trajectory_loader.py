


from collections import defaultdict
from glob import glob
from logging import warn
from typing import Sequence
import os

import numpy as np

from fgz.data_utils.data_handler import ChunkedContiguousTrajectory, get_trajectories


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

    def __init__(self, trajectories: Sequence[ChunkedContiguousTrajectory]):
        self.trajectories = trajectories

        pass

    @staticmethod
    def get_train_and_eval_loaders(dataset_path: str, train_split: float=0.8):
        train_trajectories, eval_trajectories = get_trajectories(dataset_path, train_split=train_split)

        train_loader = ContiguousTrajectoryLoader(train_trajectories)
        eval_loader = ContiguousTrajectoryLoader(eval_trajectories)

        return train_loader, eval_loader