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

        self.trajectory_loader = ContiguousTrajectoryDataLoader(dataset_path)
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


# @ray.remote
# class MultiProcessXIRLDataHandler:
#     def __init__(
#             self, dataset_path: str, device, num_workers: int=4,
#     ):
# 
#         self.handlers = []
#         for _ in range(num_workers):
#             handler = XIRLDataHandler.remote(dataset_path, device)
#             self.handlers.append(handler)
# 
#     def sample_pair(self):
#         # should trigger all handlers to begin getting their pair samples
# 
#         self.all_pairs = []
#         for handler in self.handlers:
#             self.all_pairs.append(handler.sample_pair.remote())
#         
#         while True:
#             ready_ids, _remaining_ids = ray.wait(self.all_pairs,timeout=0)
# 
#             if len(ready_ids) > 0:
#                 return ready_ids[0]
