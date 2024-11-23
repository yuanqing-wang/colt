import torch
import dgl
from dgl import DGLGraph
import math

class GCN(torch.nn.Module):
    """Graph Convolutional Networks. https://arxiv.org/abs/1609.02907

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = GCN(10, 20)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)
    
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).to(h.device).unsqueeze(1)
        h = h * norm
        h = h @ self.fc.weight.swapaxes(-1, -2)
        parallel = h.shape[0] != g.number_of_nodes()
        
        if parallel:
            h = h.swapaxes(0, -2)
            
        g.ndata["h"] = h
        g.update_all(
            dgl.function.copy_u("h", "m"),
            dgl.function.sum("m", "h"),
        )
        h = g.ndata.pop("h")
        if parallel:
            h = h.swapaxes(0, -2)
        h = h * norm
        return h
    
class GAT(torch.nn.Module):
    """Graph Attention Networks. https://arxiv.org/abs/1710.10903

    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self._in_feats, self._out_feats = in_feats, out_feats
        self.fc = torch.nn.Linear(in_feats, out_feats, bias=False)
        self.fc_edge_left = torch.nn.Linear(out_feats, 1, bias=False)
        self.fc_edge_right = torch.nn.Linear(out_feats, 1, bias=False)

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats
    
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        """Forward pass."""
        g = g.local_var()
        h = h @ self.fc.weight.swapaxes(-1, -2)
        e_left, e_right = h @ self.fc_edge_left.weight.swapaxes(-1, -2), h @ self.fc_edge_right.weight.swapaxes(-1, -2)

        parallel = h.shape[0] != g.number_of_nodes()
        if parallel:
            h = h.swapaxes(0, -2)
            e_left = e_left.swapaxes(0, -2)
            e_right = e_right.swapaxes(0, -2)
        
        g.ndata["h"] = h
        g.ndata["e_left"], g.ndata["e_right"] = e_left, e_right
        g.apply_edges(dgl.function.u_add_v("e_left", "e_right", "e"))
        g.edata["e"] = torch.nn.functional.leaky_relu(g.edata["e"], negative_slope=0.2)
        from dgl.nn.functional import edge_softmax
        e = edge_softmax(g, g.edata["e"])
        g.edata["e"] = e
        g.update_all(
            dgl.function.u_mul_e("h", "e", "m"),
            dgl.function.sum("m", "h"),
        )
        h = g.ndata.pop("h")
        if parallel:
            h = h.swapaxes(0, -2)
        return h


    