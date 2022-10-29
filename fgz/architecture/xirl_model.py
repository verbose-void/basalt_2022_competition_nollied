
from vpt.agent import AGENT_RESOLUTION, MineRLAgent, resize_image
from xirl_zero.architecture.dynamics_function import DynamicsFunction

import torch
from xirl_config import XIRLConfig

from vpt.run_agent import load_agent


class XIRLModel(torch.nn.Module):

    def __init__(self, config: XIRLConfig, device=None):
        super().__init__()

        self.config = config
        self.device = device

        agent = load_agent(config.model_path, config.weights_path, device=device)

        self.img_preprocess = agent.policy.net.img_preprocess
        self.img_process = agent.policy.net.img_process

        params = list(self.parameters())
        num_unfrozen = config.num_unfrozen_layers
        c = 0
        for param in reversed(params):
            if c >= num_unfrozen:
                param.requires_grad = False
            c += 1
        print(f"Unfrozen: {num_unfrozen}/{c}")

    def prepare_observation(self, minerl_obs):
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        img = torch.from_numpy(agent_input)
        return img

    def embed(self, frames):
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        elif frames.dim() != 4:
            raise NotImplementedError(frames.shape)

        x = self.img_preprocess(frames).unsqueeze(0)  # ficticious time-dimension
        x = self.img_process(x)
        x = x[0]  # remove time dim
        return x