import minerl
import gym
import torch
import torch.nn.functional as F

from tqdm import tqdm

from fractal_zero.search.fmc import FMC
from fractal_zero.data.tree_sampler import TreeSampler

from vpt.agent import MineRLAgent

from fgz.data_utils.data_handler import DataHandler

class FGZTrainer:

    def __init__(
        self,
        minerl_env: gym.Env,
        agent: MineRLAgent,
        fmc: FMC,
        data_handler: DataHandler,
        dynamics_function_optimizer: torch.optim.Optimizer,
        unroll_steps: int=8,
    ):
        self.minerl_env = minerl_env
        self.agent = agent
        self.data_handler = data_handler

        # TODO: create dynamics function externally as the vectorized environment inside FMC.
        # self.dynamics_function = DynamicsFunction(
        #     state_embedding_size=state_embedding_size,
        #     discriminator_classes=2,
        # )
        self.fmc = fmc
        self.unroll_steps = unroll_steps

        self.dynamics_function_optimizer = dynamics_function_optimizer

    def get_fmc_trajectory(self, root_embedding):
        self.fmc.vec_env.set_all_states(root_embedding.squeeze())
        self.fmc.reset()

        self.fmc.simulate(self.unroll_steps)

        self.tree_sampler = TreeSampler(self.fmc.tree, sample_type="best_path")
        observations, actions, _, confusions = self.tree_sampler.get_batch()
        assert len(observations) == len(actions) == len(confusions)
        return observations, actions, confusions

    def get_fmc_loss(self, fmc_root_embedding):
        fmc_obs, fmc_acts, fmc_confusions = self.get_fmc_trajectory(fmc_root_embedding)

        fmc_confusions = [c.unsqueeze(0) for c in fmc_confusions[1:]]
        fmc_confusions = torch.cat(fmc_confusions)
        fmc_discriminator_targets = torch.zeros_like(fmc_confusions)

        return F.mse_loss(fmc_confusions, fmc_discriminator_targets)

    def get_expert_loss(self, window):
        loss = 0.0
        for frame, state_embedding, action in window:
            # NOTE: FMC may decide not to go the full depth, meaning it could return observations/actions that are
            # not as long as the original window. for that case, we will shorten the window here.
            # steps = len(fmc_obs)
            # window = full_window[:steps]
            # assert len(window) == len(fmc_obs)

            # NOTE: the first value in the confusions list will be 0 or None, so we should ignore it.
            # this is because the first value corresponds to the root of the search tree (which doesn't
            # have a reward). 
            break
        return loss

    def train_trajectory(self, use_tqdm: bool=False):
        """
        2 trajectories are gathered:
            1.  From expert dataset, where the entire trajectory is lazily loaded as a contiguous piece.
            2.  Each state in the expert trajectory is embedded using a pretrained agent model. FMC runs a simulation for
                every embedded state from the expert trajectory/agent while trying to maximize a specific discriminator
                logit (to confuse the discriminator into believing it's actions came from the expert dataset).
        
        Then, the discriminator is trained to recognize the differences between true expert trajectories and trajectories
        resulting from exploiting the discriminator's confusion.
        """

        self.dynamics_function_optimizer.zero_grad()

        self.current_trajectory_window = self.data_handler.sample_single_trajectory()
        
        it = tqdm(self.current_trajectory_window, desc="Training on Trajectory", disable=not use_tqdm)
        for full_window in it:
            first_embedding = full_window[0][1]
            fmc_loss = self.get_fmc_loss(first_embedding)

            expert_loss = self.get_expert_loss(full_window)

            print(fmc_loss, expert_loss)
            loss = fmc_loss + expert_loss

            # TODO: should we backprop after the full trajectory's losses have been calculated? or should we do it each window?
            loss.backward()
            self.dynamics_function_optimizer.step()
            break

        # unroller = ExpertDatasetUnroller(self.agent, window_size=self.unroll_steps + 1)
        # for expert_sequence in unroller:
        #     start_embedding, _ = expert_sequence[0]  # "representation function"

        #     fmc_dynamics_embeddings, fmc_actions = self.get_fmc_trajectory(
        #         start_embedding, 
        #         max_steps=self.unroll_steps
        #     )
        #     expert_embeddings, expert_actions = unroller.decompose_window(expert_sequence[1:])
