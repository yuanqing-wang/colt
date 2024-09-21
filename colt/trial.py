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
            self.past, self.future = self.acquisition.step(
                past=self.past,
                future=self.future,
            )
            
    @property
    def trajectory(self):
        return np.array(
            [molecule.y for molecule in self.past]
        )
        
        
        
    