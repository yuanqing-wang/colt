from typing import List
from .molecule import Molecule, assay
from .llm import pick, tournament

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
    



        
        
    