import torch
import torch.nn as nn
import torch.distributions as D

from .mlp import MLP

class DenseDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        hidden_layers,
        layer_norm=nn.LayerNorm,
        activation=nn.ReLU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, features):
        y = self.model.forward(features)
        return y

    def loss(self, prediction, target):
        loss = self.criterion(prediction, target)
        return loss

class DenseCategoricalDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        num_channels,
        num_classes,
        hidden_dim,
        hidden_layers,
        layer_norm=nn.LayerNorm,
        activation=nn.ReLU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.out_dim = num_channels * num_classes

        self.model = MLP(in_dim, num_channels * num_classes, hidden_dim, hidden_layers, layer_norm, activation)
        self.criterion = nn.CrossEntropyLoss()

    def set_criterion_weights(self, weights):
        assert weights.shape[0] == self.num_classes, "weight dim should match num classes"
        self.criterion.weight = weights

    def forward(self, features):
        y = self.model.forward(features)
        y = y.reshape(-1, self.num_channels, self.num_classes)

        return y

    def loss(self, prediction, target):
        loss = self.criterion(prediction, target)
        return loss


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_channels = 3,
        depth: int = 32,
        activation = nn.ELU
    ):
        super().__init__()
        self.in_dim = in_dim

        #for 820x410 images
        layers = [
            nn.Linear(in_dim, 4096),
            nn.Unflatten(-1, (128, 4, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 10), stride=3), #kinda whacky kernel size
            activation(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=5),
            activation(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2),
        ]

        #xavier initialization
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y

    def loss(self, prediction, target):
        return 0.5 * torch.square(prediction - target).sum([-1, -2, -3])