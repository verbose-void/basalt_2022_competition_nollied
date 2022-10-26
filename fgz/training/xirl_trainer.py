from re import I
from typing import List
import ray
from vpt.agent import MineRLAgent
from fgz.architecture.dynamics_function import DynamicsFunction

from fgz.data_utils.xirl_data import MultiProcessXIRLDataHandler, XIRLDataHandler
from vpt.run_agent import load_agent
from xirl_config import XIRLConfig

import wandb

import torch
import torch.nn.functional as F


class XIRLModel(torch.nn.Module):

    def __init__(self, config: XIRLConfig, device=None):
        super().__init__()

        self.config = config

        agent = load_agent(config.model_path, config.weights_path, device=device)

        self.img_preprocess = agent.policy.net.img_preprocess
        self.img_process = agent.policy.net.img_process

    def embed(self, frames):
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        elif frames.dim() != 4:
            raise NotImplementedError(frames.shape)

        x = self.img_preprocess(frames).unsqueeze(0)  # ficticious time-dimension
        x = self.img_process(x)
        x = x[0]  # remove time dim
        return x

# @ray.remote
class XIRLTrainer:
    # def __init__(self, dataset_path: str, model_path: str, weights_path: str):
    def __init__(self, config: XIRLConfig):
        self.config = config

        if config.force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # data_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data_device = torch.device("cpu")

        assert len(self.config.enabled_tasks) == 1, "XIRL only supports single tasks currently."
        dataset_path = self.config.dataset_paths[0]

        # self.data_handler = XIRLDataHandler.remote(
        #     dataset_path, device=data_device
        # )
        self.data_handler = MultiProcessXIRLDataHandler.remote(
            dataset_path, device=data_device, num_workers=self.config.data_workers,
        )

        print("Model is using device", self.device)
        self.model = XIRLModel(self.config, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5, betas=(0.99, 0.999))

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
    ):
        expanded_frame_embedding = frame_embedding.expand(
            len(other_embeddings), -1
        )

        # l2 similarity
        similarity = -torch.norm(expanded_frame_embedding - other_embeddings, dim=1)
        similarity /= self.config.temperature

        alpha_k = torch.softmax(similarity, dim=0)

        return alpha_k


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

        bs = self.config.embed_batch_size

        i = 0
        while i < len(t):
            x = t[i:i+bs]
            batch = x.to(self.device)
            embedded[i:i+bs] = self.model.embed(batch)

            i += len(x)

        return embedded.float()

    def train_on_pair(self):
        assert self.config.num_frames_per_pair % self.config.batch_size == 0

        print("ray.getting the asynchronously gathered trajectory pair...")
        (self.t0, self.a0), (self.t1, self.a1) = ray.get(ray.get(self.next_data))

        # self.t0.requires_grad = True
        # self.t1.requires_grad = True

        # latency hidden data reading
        self.get_next_data()

        # TODO: start another process for gathering the next video to hide latency.
        nbytes = self.get_nbytes_stored()
        print("bytes for the pair of trajectories:", nbytes)

        print(self.t0.shape, self.t1.shape)

        self.model.train()
        with torch.no_grad():
            self.embedded_t0 = self.embed_trajectory(self.t0)
            self.embedded_t1 = self.embed_trajectory(self.t1)

        print(self.embedded_t0.shape)

        # self.dynamics_function.train()

        max_index = len(self.t0)

        num_frames = min(max_index, self.config.num_frames_per_pair)
        num_batches = num_frames // self.config.batch_size

        # num_batches = len(self.t0) // self.config.batch_size
        for _ in range(num_batches):
            self.optimizer.zero_grad()

            # total_loss = 0

            index_preds = []
            index_logits = []

            chosen_indices = torch.randint(
                low=0, high=max_index, size=(self.config.batch_size,)
            )
            frames = self.t0[chosen_indices].to(self.device)
            
            target_indices = chosen_indices.float().to(self.device)

            embeddings = self.model.embed(frames)

            # TODO: vectorize better
            for i in range(self.config.batch_size):
                # pick random frame in t0
                # self.chosen_frame_index = torch.randint(
                #     low=0, high=len(self.t0), size=(1,)
                # ).item()

                # # embed again *with gradient*
                # self.ui = self.model.embed(self.t0[self.chosen_frame_index].to(self.device))

                # self.chosen_frame_index = chosen_indices[i]
                self.ui = embeddings[i]

                # calculate cycle MSE loss
                alpha_k = self.soft_nearest_neighbor(
                    self.ui, self.embedded_t1
                )
                v_squiggly = torch.sum(alpha_k.unsqueeze(-1) * self.embedded_t1, dim=0)

                beta = self.soft_nearest_neighbor(
                    v_squiggly, self.embedded_t0
                )

                frame_mult = torch.arange(
                    start=1, end=len(beta) + 1, dtype=float, device=beta.device
                ).float()# / len(beta)
                mu = torch.matmul(frame_mult, beta) # / len(beta)

                index_preds.append(mu)
                index_logits.append(beta)

                # index_targets.append(self.chosen_frame_index)

                # print(mu, self.chosen_frame_index)
                # print(mu / max_index, self.chosen_frame_index / max_index)

                # divide both by total num frames to make indices in more reasonable range
                # mu /= total_frames
                # t = self.chosen_frame_index# / len(beta)
                # print(mu, t, len(beta))

                # cycle_loss = (1 + ((mu - t) / max_index)) ** 2
                # self_consistency_loss = self.unroll()

                # print(cycle_loss, total_loss)

                # print(cycle_loss.item(), self_consistency_loss.item())
                # total_loss += cycle_loss

            index_preds = torch.stack(index_preds)

            unnormalized_mse = F.mse_loss(index_preds, target_indices)
            normalized_mse = F.mse_loss(index_preds / max_index, target_indices / max_index)

            index_logits = torch.stack(index_logits)

            # cross_entropy_labels = F.one_hot(chosen_indices.to(self.device), num_classes=max_index, dtype=float)
            cross_entropy_labels = target_indices.long()
            ce_loss = F.cross_entropy(index_logits, cross_entropy_labels) / torch.log(torch.tensor(max_index))
            
            use_mse = True
            if use_mse:
                effective_loss = normalized_mse
            else:
                effective_loss = ce_loss

            stats = {
                "loss": effective_loss.item(),
                "metrics/video_length": max_index,
                "metrics/cross_entropy": ce_loss.item(),
                "metrics/normalized_mse": normalized_mse.item(),
                "metrics/unnormalized_mse": unnormalized_mse.item(),
                "metrics/data_bytes": nbytes,
            }

            if self.config.use_wandb:
                wandb.log(stats)

            if self.config.verbose:
                print("---------------")
                print(stats)

            effective_loss.backward()
            self.optimizer.step()

    def get_nbytes_stored(self):
        nbytes0 = sum([e.nelement() * e.element_size() for e in self.t0])
        nbytes1 = sum([e.nelement() * e.element_size() for e in self.t1])
        return nbytes0 + nbytes1
