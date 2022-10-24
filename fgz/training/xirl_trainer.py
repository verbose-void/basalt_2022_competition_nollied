from re import I
from typing import List
import ray
from vpt.agent import MineRLAgent
from fgz.architecture.dynamics_function import DynamicsFunction

from fgz.data_utils.xirl_data import XIRLDataHandler
from vpt.run_agent import load_agent
from xirl_config import XIRLConfig

import wandb

import torch
import torch.nn.functional as F


class XIRLModel(torch.nn.Module):

    def __init__(self, config: XIRLConfig):
        super().__init__()

        self.config = config

        agent = load_agent(config.model_path, config.weights_path)

        self.img_preprocess = agent.policy.net.img_preprocess
        self.img_process = agent.policy.net.img_process

    def forward(self, frames):
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        elif frames.dim() != 4:
            raise NotImplementedError(frames.shape)

        x = self.img_preprocess(frames).unsqueeze(0)  # ficticious time-dimension
        x = self.img_process(x)
        x = x[:, 0]  # remove time dim
        return x


# @ray.remote
class XIRLTrainer:
    # def __init__(self, dataset_path: str, model_path: str, weights_path: str):
    def __init__(self, config: XIRLConfig):
        self.config = config

        # TODO: config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # data_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data_device = torch.device("cpu")

        assert len(self.config.enabled_tasks) == 1, "XIRL only supports single tasks currently."
        dataset_path = self.config.dataset_paths[0]
        self.data_handler = XIRLDataHandler.remote(
            dataset_path, config.model_path, config.weights_path, device=data_device
        )

        self.model = XIRLModel(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

        # NOTE: we can't use the same agent without more complicated thread-safeness code.
        # self.agent = load_agent(model_path, weights_path)

        # self.dynamics_function = DynamicsFunction(embedder_layers=4, state_embedding_size=2048).to(device)
        # self.optimizer = torch.optim.Adam(self.dynamics_function.parameters(), lr=config.learning_rate, weight_decay=1e-5)

        # self.data_handler = XIRLDataHandler(
        #     dataset_path, self.agent, self.dynamics_function
        # )

        self.get_next_data()

    def soft_nearest_neighbor(
        self, frame_embedding: torch.Tensor,
        other_embeddings: List[torch.Tensor],
        return_similarity_vector: bool,
    ):
        # TODO: optimize this!
        soft_v = 0

        # similarity_vector = torch.zeros(size=(len(other_embeddings),), dtype=float)

        expanded_frame_embedding = frame_embedding.unsqueeze(0).expand(
            len(other_embeddings), -1
        )

        # l2 distance
        exp_norm = torch.exp(
          -torch.norm(expanded_frame_embedding - other_embeddings, dim=1)
        )

        # cosine similarity
        # exp_norm = torch.exp(F.cosine_similarity(expanded_frame_embedding, other_embeddings, dim=1))

        alpha_k = exp_norm / exp_norm.sum()

        if return_similarity_vector:
            return alpha_k

        return torch.matmul(alpha_k, other_embeddings)


    def unroll(self):
        unroll_steps = 8

        self_consistency = 0.0

        s, e = self.chosen_frame_index, self.chosen_frame_index + unroll_steps
        it = zip(self.t0[s:e], self.a0[s:e])
        unroll_state = None
        for actual_state, action in it:

            # first state should be the actual state.
            if unroll_state is None:
                unroll_state = actual_state
            else:
                unroll_state = unroll_state.squeeze()

            unroll_state = self.dynamics_function.forward_action(
                unroll_state.unsqueeze(0), action, use_discrim=False
            )

            # TODO: calculate distance between unroll state and actual state. that is the "self consistency loss"
            self_consistency += (unroll_state - actual_state) ** 2

        # print(self_consistency)

        return torch.mean(self_consistency)

    def get_next_data(self):
        print("Asynchronously gathering the next trajectory pair...")
        self.next_data = self.data_handler.sample_pair.remote()

    def embed_trajectory(self, t):
        embedded = torch.zeros(size=(len(t), 2048), device=self.device, dtype=float)

        print(embedded.shape)

        bs = self.config.batch_size

        i = 0
        while i < len(t):

            x = t[i:i+bs]
            batch = x.to(self.device)
            embedded[i:i+bs] = self.model(batch)

            print(i, batch.shape, embedded.mean())
            i += len(x)
        exit()

    def train_on_pair(self):
        assert self.config.num_frames_per_pair % self.config.batch_size == 0

        print("ray.getting the asynchronously gathered trajectory pair...")
        (self.t0, self.a0), (self.t1, self.a1) = ray.get(self.next_data)

        # self.t0.requires_grad = True
        # self.t1.requires_grad = True

        # latency hidden data reading
        self.get_next_data()

        # TODO: start another process for gathering the next video to hide latency.
        nbytes = self.get_nbytes_stored()
        print("bytes for the pair of trajectories:", nbytes)

        print(self.t0.shape, self.t1.shape)

        self.model.eval()
        with torch.no_grad():
            self.embedded_t0 = self.embed_trajectory(self.t0)
            # self.embedded_t0 = self.model(self.t0)
            # self.embedded_t1 = self.model(self.t1)

        print(self.embedded_t0.shape, self.embedded_t1.shape)
        exit()


        # self.dynamics_function.train()

        num_batches = self.config.num_frames_per_pair // self.config.batch_size
        for _ in range(num_batches):
            self.optimizer.zero_grad()

            total_loss = 0

            for _ in range(self.config.batch_size):
                # pick random frame in t0
                self.chosen_frame_index = torch.randint(
                    low=0, high=len(self.t0), size=(1,)
                ).item()

                self.ui = self.t0[self.chosen_frame_index]

                self.model(self.ui)
                exit()

                # calculate cycle MSE loss
                v_squiggly = self.soft_nearest_neighbor(
                    self.ui, self.t1, return_similarity_vector=False
                )
                beta = self.soft_nearest_neighbor(
                    v_squiggly, self.t0, return_similarity_vector=True
                )

                frame_mult = torch.arange(
                    start=1, end=len(beta) + 1, dtype=float, device=beta.device
                ).float() / len(beta)
                mu = torch.matmul(frame_mult, beta) # / len(beta)

                # divide both by total num frames to make indices in more reasonable range
                # mu /= total_frames
                t = self.chosen_frame_index / len(beta)
                # print(mu, t, len(beta))

                cycle_loss = (mu - t) ** 2
                # self_consistency_loss = self.unroll()

                # print(cycle_loss.item(), self_consistency_loss.item())
                total_loss += cycle_loss

            tcc_loss = total_loss / self.config.batch_size
            if self.config.use_wandb:
                wandb.log({
                    "tcc_loss": tcc_loss,
                    "data_bytes": nbytes,
                })

            # print("avg loss", tcc_loss.item())
            tcc_loss.backward()
            self.optimizer.step()

    def get_nbytes_stored(self):
        nbytes0 = sum([e.nelement() * e.element_size() for e in self.t0])
        nbytes1 = sum([e.nelement() * e.element_size() for e in self.t1])
        return nbytes0 + nbytes1
