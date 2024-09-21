from typing import List
import lightning as pl
import torch
import pyro
import dgl

from ..acquisition import Acquisition
from ..molecule import Molecule


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
        super().__init__()
        self.model = model
        self.acquisition_function = acquisition_function
        
    def train(self, past):
        if hasattr(self.model, "_init"):
            self.model._init(past.serve_graphs())        
        self.model.train()
        past = past.serve_graphs()
        trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False)
        trainer.fit(self.model, past)
        self.model.eval()
        
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        self.train(past)
        g, h, _ = next(iter(future.serve_graphs()))
        distributions = self.model(g, h)
        scores = self.acquisition_function(distributions).flatten()
        pick = torch.argmax(scores)
        pick = future[pick]
        return pick
        
        
    
        
    