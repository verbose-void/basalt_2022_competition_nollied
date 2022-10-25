import torch


class CartpolePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),  # keep MSE from exploding
        )

    def forward(self, observations, with_randomness: bool = False):
        observations = torch.tensor(observations).float()

        y = self.net(observations)

        if with_randomness:
            # center = embeddings.std()
            center = y.var()
            centered_uniform_noise = (torch.rand_like(y) * center) - (center / 2)
            y += centered_uniform_noise

        return y

    def parse_actions(self, actions):
        actions = torch.where(actions > 0.5, 1, 0).flatten()
        l = actions.tolist()
        if len(l) == 1:
            return l[0]
