from vpt.agent import MineRLAgent
from vpt.run_agent import load_agent

from fgz.architecture.dynamics_function import DynamicsFunction
import torch
import ray

from fgz.data_utils.data_handler import (
    ContiguousTrajectory,
    ContiguousTrajectoryDataLoader,
)


@ray.remote
class XIRLDataHandler:
    def __init__(
            self, dataset_path: str, device, # dynamics_function: DynamicsFunction
    ):
        # self.agent = agent
        # self.agent = load_agent(model_path, weights_path, device=device)  # TODO: should we use GPU or force CPU?

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

        embeddings = []
        actions = []

        with torch.no_grad():
            for i, (frame, action) in enumerate(trajectory):
                # obs = {"pov": frame}

                # embedding = self.agent.forward_observation(
                #     obs, return_embedding=True
                # ).squeeze(0)

                # TODO: maybe make use of the contiguous window and unroll steps?
                # embedding = self.dynamics_function.forward_action(
                    # agent_embedding, action, use_discrim=False
                # )
                # embedding = embedding.flatten()

                embedding = torch.tensor(frame, device=self.device)
                embeddings.append(embedding.float())
                actions.append(action)

                if max_frames is not None and i >= max_frames:
                    break

        # assert len(embeddings) == len(trajectory), f"Got {len(embeddings)}, {len(trajectory)}"
        embeddings = torch.stack(embeddings)
        embeddings.requires_grad = True
        # print("finished loading")
        return embeddings, actions

    def sample_pair(self, max_frames: int = None):
        t0 = self.trajectory_loader.sample()
        t1 = self.trajectory_loader.sample()

        if t0.uid == t1.uid:
            # try again if they're the same.
            return self.sample_pair()

        return (
            self.embed_trajectory(t0, max_frames=max_frames),
            self.embed_trajectory(t1, max_frames=max_frames),
        )


@ray.remote
class MultiProcessXIRLDataHandler:
    def __init__(
            self, dataset_path: str, device, num_workers: int=4,
    ):

        self.num_workers = num_workers

        self.handlers = []
        self.tasks = []
        for _ in range(num_workers):
            handler = XIRLDataHandler.remote(dataset_path, device)
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

