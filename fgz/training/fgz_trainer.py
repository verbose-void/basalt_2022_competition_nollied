import minerl
import gym

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

    def get_fmc_trajectory(self, root_embedding):
        self.fmc.vec_env.set_all_states(root_embedding.squeeze())
        self.fmc.reset()

        self.fmc.simulate(self.unroll_steps)

        self.tree_sampler = TreeSampler(self.fmc.tree, sample_type="best_path")
        observations, actions, _ = self.tree_sampler.get_batch()
        assert len(observations) == len(actions)
        return observations, actions

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

        self.current_trajectory_window = self.data_handler.sample_single_trajectory()
        
        it = tqdm(self.current_trajectory_window, desc="Sliding T Window", disable=not use_tqdm)
        for full_window in it:
            for frame, state_embedding, action in full_window:
                fmc_obs, fmc_acts = self.get_fmc_trajectory(state_embedding)

                # NOTE: FMC may decide not to go the full depth, meaning it could return observations/actions that are
                # not as long as the original window. for that case, we will shorten the window here.
                steps = len(fmc_obs)
                window = full_window[:steps]
                assert len(window) == len(fmc_obs)
                break
            break

        # unroller = ExpertDatasetUnroller(self.agent, window_size=self.unroll_steps + 1)
        # for expert_sequence in unroller:
        #     start_embedding, _ = expert_sequence[0]  # "representation function"

        #     fmc_dynamics_embeddings, fmc_actions = self.get_fmc_trajectory(
        #         start_embedding, 
        #         max_steps=self.unroll_steps
        #     )
        #     expert_embeddings, expert_actions = unroller.decompose_window(expert_sequence[1:])
        pass
