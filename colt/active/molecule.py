from dataclasses import dataclass
from typing import Optional

@dataclass
class Molecule:
    smiles: str
    graph: Optional[str] = None
    y: Optional[float] = None
    
    def featurize(self):
        from dgllife.utils import (
            smiles_to_bigraph,
            CanonicalBondFeaturizer,
            CanonicalAtomFeaturizer,
        )
                
        self.graph = smiles_to_bigraph(
            self.smiles,
            node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'),
            edge_featurizer=None,
        )
        
    def __hash__(self):
        return hash(self.smiles)
    
    def __eq__(self, other):
        return self.smiles == other.smiles
    

@dataclass
class Dataset:
    molecules: tuple[Molecule]
        
    def featurize(self):
        for molecule in self.molecules:
            molecule.featurize()
            
    def lookup(self, molecule):
        for _molecule in self.molecules:
            if _molecule == molecule:
                return _molecule
            
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, index):
        return self.molecules[index]
            
    def __sub__(self, molecule):
        return Dataset([m for m in self.molecules if m != molecule])
    
    def __add__(self, molecule):
        return Dataset(self.molecules + (molecule,))
            
def assay(
    molecule: Molecule,
    past: Dataset,
    future: Dataset,
):
    molecule = future.lookup(molecule)
    future = future - molecule
    past = past + molecule
    return past, future

    
    



        
    