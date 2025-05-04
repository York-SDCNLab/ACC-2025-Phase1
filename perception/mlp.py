import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        hidden_layers, 
        layer_norm, 
        activation=nn.ELU
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        dim = in_dim
        for i in range(hidden_layers):
            layers += [nn.Linear(dim, hidden_dim)]
            if layer_norm:
                layers += [nn.LayerNorm(hidden_dim, eps=1e-3),]

            layers += [activation()]
            dim = hidden_dim

        layers += [nn.Linear(hidden_dim, out_dim),]

        if out_dim == 1:
            layers += [nn.Flatten(0),]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y