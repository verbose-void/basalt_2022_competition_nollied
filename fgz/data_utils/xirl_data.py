
from vpt.agent import MineRLAgent
from fgz.architecture.dynamics_function import DynamicsFunction
import torch

from fgz.data_utils.data_handler import ContiguousTrajectory, ContiguousTrajectoryDataLoader


class XIRLDataHandler:

    def __init__(self, dataset_path: str, agent: MineRLAgent, dynamics_function: DynamicsFunction):
        self.agent = agent
        self.trajectory_loader = ContiguousTrajectoryDataLoader(dataset_path)
        self.dynamics_function = dynamics_function

    def embed_trajectory(self, trajectory: ContiguousTrajectory):
        # reset hidden state.
        self.agent.reset()

        embeddings = []

        for frame, action in trajectory:
            obs = {"pov": frame}

            with torch.no_grad():
                agent_embedding = self.agent.forward_observation(obs, return_embedding=True).squeeze(0)

            # TODO: maybe make use of the contiguous window and unroll steps?
            embedding = self.dynamics_function.forward_action(agent_embedding, action, use_discrim=False)
            embeddings.append(embedding)

        # assert len(embeddings) == len(trajectory), f"Got {len(embeddings)}, {len(trajectory)}"
        return embeddings

    def sample_pair(self):
        t0 = self.trajectory_loader.sample()
        t1 = self.trajectory_loader.sample()

        if t0.uid == t1.uid:
            # try again if they're the same.
            return self.sample_pair()

        self.embedded_t0 = self.embed_trajectory(t0)
        self.embedded_t1 = self.embed_trajectory(t1)
        return self.embedded_t0, self.embedded_t1

    def get_nbytes_stored(self):
        nbytes0 = sum([e.nelement() * e.element_size() for e in self.embedded_t0])
        nbytes1 = sum([e.nelement() * e.element_size() for e in self.embedded_t1])
        return nbytes0 + nbytes1
