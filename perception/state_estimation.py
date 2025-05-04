import torch
import torch.nn as nn

from core.model.rnn import GRUCellStack

class StateEstimator(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        waypoint_dim: int = 200,
        action_dim: int = 2,
        hidden_dim: int = 64,
        n_layer: int = 1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        #input_dim = int((state_dim + action_dim)*horizon_length)
        self.output_dim = state_dim

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.state_mlp = nn.Sequential(*layers)

        layers = [
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.action_mlp = nn.Sequential(*layers)

        layers = [
            nn.Linear(waypoint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.waypoint_mlp = nn.Sequential(*layers)

        layers = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.posterior_mlp = nn.Sequential(*layers)

        self.gru = GRUCellStack(hidden_dim, hidden_dim, 1, "gru")

    def forward(
        self, 
        state, 
        action, 
        #waypoints,
        in_h
    ):
        state_emb = self.state_mlp(state)
        action_emb = self.action_mlp(action)
        #waypoint_emb = self.waypoint_mlp(waypoints)
        x = state_emb + action_emb #+ waypoint_emb

        h = self.gru(x, in_h)

        y = self.posterior_mlp(h)

        return h, y