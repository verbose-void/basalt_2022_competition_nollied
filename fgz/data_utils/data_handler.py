from collections import defaultdict
import os
from glob import glob
from warnings import warn

import minerl
from typing import List, Sequence, Tuple

from fgz.data_utils.data_loader import get_json_length_without_null_actions, trajectory_generator
import torch

from vpt.agent import AGENT_RESOLUTION, MineRLAgent, resize_image

import numpy as np
import cv2


class ContiguousTrajectory:
    """Note: when iterating this class, only 1 frame will be yielded at a time.
    For a window/batch, use `ContiguousTrajectoryWindow`.
    """

    def __init__(self, video_path: str, json_path: str, uid: str, task_id: int = None):
        self.video_path = video_path
        self.json_path = json_path
        self.uid = uid
        self.task_id = task_id

    def __len__(self):
        # with open(self.json_path) as json_file:
        #     return len(json_file.readlines())

        return get_json_length_without_null_actions(self.json_path)

    def __str__(self) -> str:
        return f"T({self.uid})"

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self, start_frame: int = None):
        return trajectory_generator(
            self.video_path, self.json_path, start_frame=start_frame
        )


class ChunkedContiguousTrajectory:

    def __init__(self, trajectory_prefix: str, clip_date_times: Sequence[Tuple], uid: str, task_id: int = None):

        self.video_paths = []
        self.json_paths = []
        for date, time in clip_date_times:
            path_pref = f"{trajectory_prefix}-{date}-{time}"

            self.video_paths.append(f"{path_pref}.mp4")
            self.json_paths.append(f"{path_pref}.jsonl")

        self.uid = uid
        self.task_id = task_id

        assert len(self.video_paths) == len(self.json_paths)

        self.contiguous_clips: Sequence[ContiguousTrajectory] = []
        for video_path, json_path in zip(self.video_paths, self.json_paths):

            if not os.path.exists(video_path) or not os.path.exists(json_path):
                raise ValueError

            clip = ContiguousTrajectory(video_path, json_path, uid=uid, task_id=task_id)
            self.contiguous_clips.append(clip)

        # TODO: if any of them are corrupted or don't exist, raise an exception. this chunked clip object should not persist
        
        self.num_clips = len(self.contiguous_clips)

    def __len__(self):
        return sum([len(clip) for clip in self.contiguous_clips])

    def __str__(self) -> str:
        return f"CT({self.uid}, nclips={self.num_clips})"

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self, start_frame: int = None):
        if start_frame is not None:
            raise NotImplementedError
        else:
            self._current_clip = 0

        self._clip_iter = iter(self.contiguous_clips[self._current_clip])
        
        # return trajectory_generator(
        #     self.video_path, self.json_path, start_frame=start_frame
        # )
        return self

    def __next__(self):
        """Chain together all of the contiguous clip iterators."""

        try:
            data = self._clip_iter.__next__()
            return data
        except StopIteration:
            if self._current_clip >= self.num_clips - 1:
                raise StopIteration

            self._current_clip += 1
            self._clip_iter = iter(self.contiguous_clips[self._current_clip])

            return self.__next__()

    def get_last_frame(self):
        clip = self.contiguous_clips[-1]
        video = cv2.VideoCapture(clip.video_path)
        last_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        video.set(1, last_frame_num)
        ret, frame = video.read()

        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frame = resize_image(frame, AGENT_RESOLUTION)

        return frame

class ContiguousTrajectoryWindow:
    def __init__(
        self,
        trajectory: ContiguousTrajectory,
        agent: MineRLAgent,
        frames_per_window: int = 4,
        allow_agent_gradients: bool = False,
        use_random_subsection: bool = False,
        num_strides: int = 8,
    ):
        self.trajectory = trajectory
        self.agent = agent
        self.frames_per_window = frames_per_window
        self.allow_agent_gradients = allow_agent_gradients

        # TODO: explain
        self.use_random_subsection = use_random_subsection
        self.num_strides = num_strides  # TODO: stride length

        assert self.frames_per_window > 0
        assert self.num_strides > 0

        if self.use_random_subsection:
            required_num_frames = self.frames_per_window + self.num_strides
            self.start = np.random.randint(
                low=0, high=len(self.trajectory) - required_num_frames
            )
            self.end = self.start + required_num_frames
        else:
            self.start = 0
            self.end = len(self.trajectory)

        assert self.end > self.start

    @property
    def uid(self):
        return self.trajectory.uid

    @property
    def task_id(self):
        return self.trajectory.task_id

    def num_frames(self):
        return self.end - self.start

    def __len__(self):
        return self.num_strides

    def __iter__(self):
        self._num_returned_windows = 0
        self._trajectory_iterator = self.trajectory.__iter__(start_frame=self.start)
        self.window = []
        return self

    def __next__(self, populating: bool = False):
        if self._num_returned_windows >= self.num_strides:
            self._trajectory_iterator = None
            raise StopIteration

        is_first = self._num_returned_windows == 0 and not populating

        frame, action = self._trajectory_iterator.__next__()

        # TODO: this doesn't take into consideration batching w.r.t trajectories!
        obs = {"pov": frame}
        state_embedding = self.agent.forward_observation(obs, return_embedding=True)

        if not self.allow_agent_gradients:
            state_embedding = state_embedding.detach()

        self.window.append((frame, state_embedding, action))
        if len(self.window) > self.frames_per_window:
            self.window.pop(0)

        if is_first:
            # let the window fully populate
            for _ in range(self.frames_per_window - 1):
                self.__next__(populating=True)

        if populating:
            return

        if len(self.window) != self.frames_per_window:
            raise ValueError(
                f"Unexpected window size. Got {len(self.window)}, expected: {self.frames_per_window}"
            )

        self._num_returned_windows += 1
        return self.window


def get_trajectories(dataset_path: str, for_training: bool, train_split: float):
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

    # sort and gather the trajectories into a single class
    trajectories: Sequence[ChunkedContiguousTrajectory] = []

    # training gets the first section, validation gets last
    it = list(sorted(full_trajectory_ids.keys()))
    it = it[:max_train_idx] if for_training else it[max_train_idx:]

    for trajectory_prefix in it:
        date_times = full_trajectory_ids[trajectory_prefix]

        sorted_date_times = list(sorted(date_times))
        # print(trajectory_prefix, sorted_date_times)

        try:
            chunked_traj = ChunkedContiguousTrajectory(trajectory_prefix, sorted_date_times, trajectory_prefix)#, task_id=task_id)
            trajectories.append(chunked_traj)
        except ValueError:
            warn(f"Missing video/json path! Skipping... {trajectory_prefix} {sorted_date_times}")
            continue

    return trajectories


class ContiguousTrajectoryDataLoader:
    def __init__(self, dataset_path: str, task_id: int = None, minimum_steps: int = 64, is_train: bool=True, train_split: float=0.8):
        self.dataset_path = dataset_path
        self.task_id = task_id
        self.minimum_steps = minimum_steps
        self.is_train = is_train

        self.trajectories = get_trajectories(dataset_path, for_training=is_train, train_split=train_split)
        # create ContiguousTrajectory objects for every mp4/json file pair.
        # self.trajectories = []
        # for unique_id in sorted(unique_ids):
        #     video_path = os.path.abspath(
        #         os.path.join(self.dataset_path, unique_id + ".mp4")
        #     )
        #     json_path = os.path.abspath(
        #         os.path.join(self.dataset_path, unique_id + ".jsonl")
        #     )

        #     if not os.path.exists(video_path) or not os.path.exists(json_path):
        #         warn(f"Skipping {unique_id}...")
        #         continue

        #     t = ContiguousTrajectory(video_path, json_path, unique_id, task_id)
        #     self.trajectories.append(t)

        #     if max_num_trajectories and len(self.trajectories) >= max_num_trajectories:
        #         break 


    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        # trajectories are always randomly shuffled. frame ordering remains in-tact.
        self.permutation = torch.randperm(len(self))
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration

        t_index = self.permutation[self._iter].item()
        yield self.trajectories[t_index]
        self._iter += 1

    def sample(self) -> ContiguousTrajectory:
        trajectory_index = np.random.randint(low=0, high=len(self))
        t = self.trajectories[trajectory_index]

        # remove the trajectory and pick a new sample if it's not long enough.
        t_len = len(t)
        if t_len < self.minimum_steps:
            self.trajectories.pop(trajectory_index)
            warn(
                f"Removing trajectory from the dataset. It's length was too short ({t_len} < {self.minimum_steps}). Now there are {len(self.trajectories)} left."
            )
            return self.sample()

        return t

    def __str__(self):
        return f"ContiguousTrajectoryDataLoader(n={len(self)}, {self.dataset_path})"


class DataHandler:
    def __init__(
        self, dataset_paths: List[str], agent: MineRLAgent, frames_per_window: int
    ):
        self.dataset_paths = dataset_paths
        self.frames_per_window = frames_per_window
        self.loaders = [
            ContiguousTrajectoryDataLoader(path, task_id)
            for task_id, path in enumerate(self.dataset_paths)
        ]
        self.agent = agent

    @property
    def num_tasks(self):
        # one loader per task
        return len(self.loaders)

    def sample_trajectories_for_each_task(self):
        # from each task dataset, sample 1 trajectory.
        # TODO: maybe sample more than 1 per task
        return [loader.sample() for loader in self.loaders]

    def sample_single_trajectory(self, num_strides: int = 1):
        task_id = np.random.randint(low=0, high=self.num_tasks)
        trajectory = self.loaders[task_id].sample()
        return ContiguousTrajectoryWindow(
            trajectory,
            agent=self.agent,
            frames_per_window=self.frames_per_window,
            num_strides=num_strides,
            use_random_subsection=True,
        )


# class ExpertDatasetUnroller:
#     """Generates a window of state embeddings generated by a MineRLAgent. The agent
#     assumes that the trajectories are fed in contiguously. This class is meant to be iterated over!
#     """

#     def __init__(self, agent: MineRLAgent, window_size: int=4):
#         self.agent = agent
#         self.window_size = window_size

#     def __iter__(self):
#         self._iter = 0

#         # TODO: reset the agent's internal state and populate these with a FULL TRAJECTORY!
#         self.expert_observations = []
#         self.expert_actions = []
#         self._expert_pairs = zip(self.expert_observations, self.expert_actions)

#         self.window = []

#     def __next__(self, dont_yield: bool=False):
#         is_first = self._iter == 0
#         self._iter += 1

#         # should auto-raise StopIteration
#         expert_observation, expert_action = self._expert_pairs.__next__()

#         # precompute the expert embeddings
#         expert_embedding = self.agent.get_embedding(expert_observation)
#         self.window.append((expert_embedding, expert_action))
#         if len(self.window) > self.window_size:
#             self.window.pop(0)

#         if is_first:
#             # let the window fully populate
#             for _ in range(self.window_size - 1):
#                 self.__next__(dont_yield=True)

#         if len(self.window) != self.window_size:
#             raise ValueError(f"Unexpected window size. Got {len(self.window)}, expected: {self.window_size}")

#         if not dont_yield:
#             yield self.window

#     def decompose_window(self, window: List):
#         embeddings = []
#         actions = []
#         for embedding, action in window:
#             embeddings.append(embedding)
#             actions.append(action)
#         return embeddings, actions
