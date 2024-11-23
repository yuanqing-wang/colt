import torch

class Linear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            activation: torch.nn.Module = torch.nn.Tanh(),
            **kwargs,
    ):
        super().__init__()
        self.W0 = torch.nn.Parameter(1e-2 * torch.randn(in_features, hidden_features))
        self.B0 = torch.nn.Parameter(1e-2 * torch.randn(hidden_features))
        self.W1 = torch.nn.Parameter(1e-2 * torch.randn(hidden_features, out_features))
        self.B1 = torch.nn.Parameter(1e-2 * torch.randn(out_features))
        self.activation = activation
        self.hidden_features = hidden_features
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h @ self.W0 + self.B0
        h = self.activation(h)
        h = h @ self.W1 + self.B1
        return h