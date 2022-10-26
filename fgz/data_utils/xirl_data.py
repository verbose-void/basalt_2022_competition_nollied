from logging import warn
from vpt.agent import MineRLAgent
from vpt.run_agent import load_agent

from fgz.architecture.dynamics_function import DynamicsFunction
import torch
import ray

from fgz.data_utils.data_handler import (
    ContiguousTrajectory,
    ContiguousTrajectoryDataLoader,
)
from xirl_config import XIRLConfig


@ray.remote
class XIRLDataHandler:
    def __init__(
            self, config: XIRLConfig, dataset_path: str, device, # dynamics_function: DynamicsFunction
    ):
        # self.agent = agent
        # self.agent = load_agent(model_path, weights_path, device=device)  # TODO: should we use GPU or force CPU?

        self.config = config
        self.device = device

        # TODO: set to None!
        max_num_trajectories = None
        self.trajectory_loader = ContiguousTrajectoryDataLoader(dataset_path, max_num_trajectories=max_num_trajectories)
        # self.dynamics_function = dynamics_function

    def embed_trajectory(
        self, trajectory: ContiguousTrajectory, max_frames: int = None
    ):
        # reset hidden state.
        # self.agent.reset()

        frames = []
        actions = []

        with torch.no_grad():

            # NOTE: some frames may get corrupt during the loading process, 
            # if that occurs not all frames will be used probably. it may not
            # have a huge impact on training, but it's definitely good to be
            # aware of.

            # we can't load more frames than are available
            nframes_to_use = min(self.config.num_frames_per_trajectory_to_load, len(trajectory))

            frames_to_use = torch.round(torch.linspace(start=0, end=len(trajectory), steps=nframes_to_use)).int().tolist()

            c = 0

            last_frame = None
            for i, (frame, action) in enumerate(trajectory):
                if max_frames is not None and i >= max_frames:
                    break

                # NOTE: this is a hack -- it would be faster to not load these frames from the disk at all.
                if i not in frames_to_use:
                    continue

                # obs = {"pov": frame}

                # embedding = self.agent.forward_observation(
                #     obs, return_embedding=True
                # ).squeeze(0)

                # TODO: maybe make use of the contiguous window and unroll steps?
                # embedding = self.dynamics_function.forward_action(
                    # agent_embedding, action, use_discrim=False
                # )
                # embedding = embedding.flatten()

                frame_tensor = torch.tensor(frame, device=self.device)
                frames.append(frame_tensor.float())
                actions.append(action)

                c += 1

                last_frame = frame_tensor

            # print(frames_to_use)
            # print(len(frames_to_use))
            assert len(frames) <= len(frames_to_use)

            # ensure we always have the last frame in the sequence.
            if last_frame is not None and c < len(frames_to_use):
                frames.append(last_frame)

        # assert len(embeddings) == len(trajectory), f"Got {len(embeddings)}, {len(trajectory)}"
        frames = torch.stack(frames)
        frames.requires_grad = True
        # print("finished loading")
        return frames, actions

    def sample_pair(self, max_frames: int = None):
        t0 = self.trajectory_loader.sample()
        t1 = self.trajectory_loader.sample()

        if t0.uid == t1.uid:
            # try again if they're the same.
            return self.sample_pair()

        try:
            emb0 = self.embed_trajectory(t0, max_frames=max_frames)
            emb1 = self.embed_trajectory(t1, max_frames=max_frames)
            return emb0, emb1
        except:
            warn("Failed to embed trajectories. Trying to sample again...")
            return self.sample_pair()


@ray.remote
class MultiProcessXIRLDataHandler:
    def __init__(
            self,  config: XIRLConfig, dataset_path: str, device, num_workers: int=4,
    ):

        self.num_workers = num_workers

        self.handlers = []
        self.tasks = []
        for _ in range(num_workers):
            handler = XIRLDataHandler.remote(config, dataset_path, device)
            self.handlers.append(handler)

            # kick-start sampling
            self.tasks.append(handler.sample_pair.remote())

    def sample_pair(self):
        # should trigger all handlers to begin getting their pair samples

        assert len(self.tasks) == len(self.handlers) == self.num_workers
        
        while True:
            ready_ids, _remaining_ids = ray.wait(self.tasks, num_returns=self.num_workers, timeout=0.1)

            if len(ready_ids) > 0:
                print("num that were ready", len(ready_ids))

                ready_index = self.tasks.index(ready_ids[0])
                ready_id = self.tasks[ready_index]

                # overwrite task with new one.
                self.tasks[ready_index] = self.handlers[ready_index].sample_pair.remote()

                # move the task and corresponding handler to the back of the line.
                moving_id = self.tasks.pop(ready_index)
                moving_handler = self.handlers.pop(ready_index)
                self.tasks.append(moving_id)
                self.handlers.append(moving_handler)

                # return ray.get(ready_id)
                return ready_id

        raise ValueError("This shouldn't be possible...")

