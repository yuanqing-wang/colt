from functools import partial
import torch
import dgl
from .zoo import GCN

class Sequential(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 3,
        layer: torch.nn.Module = GCN,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                layer(
                    in_features if idx == 0 else hidden_features,
                    hidden_features,
                )
                for idx in range(depth)
            ]
        )
        self.activation = activation
        self.pool = Aggregation(hidden_features, out_features)
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = self.activation(layer(g, h))
        h = self.pool(g, h)
        return h
    
    
class Aggregation(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: torch.nn.Module = torch.nn.Tanh(),
            aggregator: torch.nn.Module = dgl.mean_nodes,
            **kwargs,
    ):
        super().__init__()
        self.W0 = torch.nn.Parameter(1e-2 * torch.randn(in_features, in_features))
        self.B0 = torch.nn.Parameter(1e-2 * torch.randn(in_features))
        self.W1 = torch.nn.Parameter(1e-2 * torch.randn(in_features, out_features))
        self.B1 = torch.nn.Parameter(1e-2 * torch.randn(out_features))
        self.activation = activation
        self.aggregator = aggregator

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            **kwargs,
    ):
        """Forward pass."""
        g = g.local_var()
        parallel = h.shape[0] != g.number_of_nodes()
        if parallel:
            h = h.swapaxes(0, -2)
        g.ndata["h"] = h
        h = self.aggregator(g, "h")
        if parallel:
            h = h.swapaxes(0, -2)
        h = h @ self.W0 + self.B0
        h = self.activation(h)
        h = h @ self.W1 + self.B1
        return h