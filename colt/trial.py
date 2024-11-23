from typing import List
import numpy as np
from .molecule import Molecule, Dataset, assay
from .acquisition import Acquisition

class Trial:
    def __init__(
        self,
        future: Dataset,
        acquisition: Acquisition,
        steps: int = 1,
    ):
        self.future = future
        self.acquisition = acquisition
        self.past = Dataset(tuple())
        self.steps = steps
    
    def blind(self):
        from random import choice
        best = choice(self.future)
        self.past, self.future = assay(best, self.past, self.future)
    
    def loop(self):
        self.blind()
        for step in range(self.steps):
            if len(self.future) == 0:
                break
            
            if max([molecule.y for molecule in self.past]) \
                > max([molecule.y for molecule in self.future]):
                break
            
            self.past, self.future = self.acquisition.step(
                past=self.past,
                future=self.future,
            )
            
    @property
    def trajectory(self):
        return np.array(
            [molecule.y for molecule in self.past]
        )
        
    @property
    def smiles_trajectory(self):
        return [molecule.smiles for molecule in self.past]
        
        
    