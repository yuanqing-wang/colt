from ..acquisition import Acquisition

import torch
import pyro
import dgl


# ==============================
# Training utility
# ==============================
def collate(past):
    past.featurize()
    g = dgl.batch([molecule.graph for molecule in past])
    h = g.ndata['h']
    y = torch.tensor([molecule.y for molecule in past]).unsqueeze(-1)
    data = [g, h, y]
    return data

# ==============================
# Acquisition functions
# ==============================
def expected_improvement(
        y: torch.Tensor,
        best: torch.Tensor = 0.0,
):
    ei = (y - best).clamp(min=0).mean(dim=0)
    return ei

def probability_of_improvement(
        y: torch.Tensor,
        best: torch.Tensor = 0.0,
):
    pi = (y > best).float().mean(dim=0)
    return pi

# ==============================
# Acquisition class
# ==============================

class BayesianAcquisition(Acquisition):
    def __init__(
            self, 
            model,
            acquisition_function: callable = expected_improvement,
        ):
        super.__init__()
        self.model = model
        
    def train(self, past):
        self.model.train(past)