import torch

from fractal_zero.models.dynamics import FullyConnectedDynamicsModel
from fractal_zero.models.prediction import FullyConnectedPredictionModel
from fractal_zero.models.representation import FullyConnectedRepresentationModel


class JointModel(torch.nn.Module):
    def __init__(
        self,
        representation_model: FullyConnectedRepresentationModel,
        dynamics_model: FullyConnectedDynamicsModel,
        prediction_model: FullyConnectedPredictionModel,
    ):
        super().__init__()

        # TODO: base classes for rep/dyn/pred models
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model

        # default on CPU
        self.to(torch.device("cpu"))

    def to(self, device):
        self.device = device
        self.representation_model = self.representation_model.to(device)
        self.dynamics_model = self.dynamics_model.to(device)
        self.prediction_model = self.prediction_model.to(device)
        return super().to(device)
