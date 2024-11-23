from typing import List
from .molecule import Molecule, assay

class Acquisition:    
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        raise NotImplementedError
    
    def step(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        best = self.pick(past, future)
        past, future = assay(best, past, future)
        return past, future
    
class RandomAcquisition(Acquisition):
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        import random
        return random.choice(future)



        
        
    