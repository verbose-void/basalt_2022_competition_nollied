from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import os

from fgz.architecture.xirl_model import XIRLModel
from fgz.data_utils.data_handler import ContiguousTrajectoryDataLoader
from fgz.data_utils.generate_xirl_targets import generate_target


MINERL_DATA_ROOT = os.getenv("MINERL_DATA_ROOT", "data/")
VPT_MODELS_ROOT = os.path.join(MINERL_DATA_ROOT, "VPT-models/")


@dataclass
class TCCConfig:
    # model_filename: str = "foundation-model-1x.model"
    model_filename: str = "foundation-model-2x.model"
    # model_filename: str = "foundation-model-3x.model"

    # weights_filename: str = "foundation-model-1x.weights"
    weights_filename: str = "rl-from-early-game-2x.weights"

    learning_rate: float = 0.001
    temperature: float = 0.1

    # the number of unfrozen modules in the representation model.
    # https://analyticsindiamag.com/what-does-freezing-a-layer-mean-and-how-does-it-help-in-fine-tuning-neural-networks/
    num_unfrozen_layers: int = 4  

    batch_size: int = 32        # gradients
    embed_batch_size: int = 32  # no gradients

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def model_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.model_filename)

    @property
    def weights_path(self) -> str:
        return os.path.join(VPT_MODELS_ROOT, self.weights_filename)

class TCCRepresentationTrainer:

    def __init__(self, config: TCCConfig):
        self.config = config

        self.model = XIRLModel(self.config, config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=0, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5, betas=(0.99, 0.999))

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

    def embed_trajectory(self, t: torch.Tensor):
        embedded = torch.zeros(size=(len(t), 2048), device=self.config.device, dtype=float)

        bs = self.config.embed_batch_size

        i = 0
        while i < len(t):
            x = t[i:i+bs]
            batch = x.to(self.config.device)
            embedded[i:i+bs] = self.model.embed(batch)

            i += len(x)

        return embedded.float()

    def _calculate_temporal_cycle_consistency(
        self, 
        embedded_t0: torch.Tensor, 
        embedded_t1: torch.Tensor, 
        chosen_frame_indices: torch.Tensor,
        embedded_chosen_frames: torch.Tensor,    
    ):

        # TODO: vectorize better

        bs = self.config.batch_size
        assert len(chosen_frame_indices) == len(embedded_chosen_frames) == bs

        max_index = len(embedded_t0)

        index_preds = []
        index_logits = []

        for i in range(bs):
            ui = embedded_chosen_frames[i]

            # calculate cycle MSE loss
            alpha_k = self.soft_nearest_neighbor(
                ui, embedded_t1
            )
            v_squiggly = torch.sum(alpha_k.unsqueeze(-1) * embedded_t1, dim=0)

            beta = self.soft_nearest_neighbor(
                v_squiggly, embedded_t0
            )

            frame_mult = torch.arange(
                start=1, end=len(beta) + 1, dtype=float, device=beta.device
            ).float()# / len(beta)
            mu = torch.matmul(frame_mult, beta) # / len(beta)

            index_preds.append(mu)
            index_logits.append(beta)

        index_preds = torch.stack(index_preds)
        index_logits = torch.stack(index_logits)
        cross_entropy_labels = chosen_frame_indices.long()
        
        stats = {
            "normalized_mse": F.mse_loss(index_preds / max_index, chosen_frame_indices / max_index),
            "unnormalized_mse": F.mse_loss(index_preds, chosen_frame_indices),
            "cross_entropy": F.cross_entropy(index_logits, cross_entropy_labels), # / torch.log(torch.tensor(max_index))
        }

        return stats

    def choose_random_frames(self, max_index: int):
        return torch.randint(
            low=0, high=max_index, size=(self.config.batch_size,)
        )

    def _calculate_loss(self, t0: torch.Tensor, t1: torch.Tensor, with_gradient: bool) -> Tuple[Dict, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            embedded_t0 = self.embed_trajectory(t0)
            embedded_t1 = self.embed_trajectory(t1)

        chosen_frame_indices = self.choose_random_frames(len(embedded_t0))

        if with_gradient:
            # embed chosen frames again, but this time with gradients
            embedded_chosen_frames = self.model.embed(t0[chosen_frame_indices].to(self.config.device))
        else:
            # don't embed again, no need to calculate gradients
            embedded_chosen_frames = embedded_t0[chosen_frame_indices]

        def _execute():
            return self._calculate_temporal_cycle_consistency(
                embedded_t0, 
                embedded_t1, 
                chosen_frame_indices.to(self.config.device),
                embedded_chosen_frames,
            )

        if with_gradient:
            stats = _execute()
        else:
            with torch.no_grad():
                stats = _execute()

        return stats, embedded_t0, embedded_t1

    def train_step(self, t0: torch.Tensor, t1: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """The representation function is trained using Temporal Cycle-Consistency (https://arxiv.org/pdf/1904.07846.pdf) 
        loss over a subset of the frames in a pair of demonstration trajectories.
        """

        self.model.train()
        self.optimizer.zero_grad()

        stats, embedded_t0, embedded_t1 = self._calculate_loss(t0, t1, with_gradient=True)

        loss = stats["normalized_mse"]

        loss.backward()
        self.optimizer.step()

        return stats, embedded_t0, embedded_t1

    @torch.no_grad()
    def eval_step(self, t0: torch.Tensor, t1: torch.Tensor):
        self.model.eval()
        stats, embedded_t0, embedded_t1 = self._calculate_loss(t0, t1, with_gradient=False)
        return stats, embedded_t0, embedded_t1

    @torch.no_grad()
    def generate_target_state(self, loader: ContiguousTrajectoryDataLoader, use_tqdm: bool=True):
        return generate_target(self.config, self.model, loader, use_tqdm=use_tqdm, device=self.config.device)