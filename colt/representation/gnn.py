from functools import partial
import torch
import dgl
from .zoo import GCN
from .representation import Representation
from .linear import Linear

class GNN(Representation):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 3,
        layer: torch.nn.Module = GCN,
        activation: torch.nn.Module = torch.nn.SiLU(),
        dropout: float = 0.5,
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
        # self.dropout = torch.nn.Dropout(dropout)
        self.pool = Aggregation(hidden_features, out_features)
        self.hidden_features = hidden_features
        self.out_features = out_features
        
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
        self.linear = Linear(in_features, out_features, in_features)
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
        h = self.linear(h)
        return h