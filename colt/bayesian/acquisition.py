from typing import List
import lightning as pl
import torch
import pyro
import dgl

from ..acquisition import Acquisition
from ..molecule import Molecule

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

def upper_confidence_bound(
        y: torch.Tensor,
        beta: float = 1.0,
):
    ucb = y.mean(dim=0) + beta * y.std(dim=0)
    return ucb

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
        x, y = past.batch()
        if hasattr(self.model, "_init"):
            self.model._init(x, y) 
        self.model.train()
        trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False, accelerator="gpu")
        trainer.fit(self.model, ((x, y) for _ in range(1)))
        self.model.eval()
        
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        self.train(past)
        x, _ = future.batch()
        distributions = self.model(x)
        scores = self.acquisition_function(distributions).flatten()
        pick = torch.argmax(scores)
        pick = future[pick]
        return pick
        
        
    
        
    