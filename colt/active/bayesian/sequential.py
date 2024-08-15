from re import L
import torch
import dgl

class Sequential(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 3,
        layer: torch.nn.Module = dgl.nn.GraphConv,
        activation: torch.nn.Module = torch.nn.SiLU(),
        pool: torch.nn.Module = dgl.nn.AveragePooling(),
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                layer(
                    in_features if idx == 0 else hidden_features,
                    hidden_features if idx != len(hidden_features) else out_features,
                )
                for idx in range(depth)
            ]
        )
        self.activation = activation
        self.pool = pool
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = self.activation(layer(g, h))
        h = self.pool(g, h)
        return h