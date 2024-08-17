from dataclasses import dataclass
from typing import Optional
import torch
import dgl

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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, molecules):
        self.molecules = tuple(molecules)
        
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
        return self.__class__([m for m in self.molecules if m != molecule])
    
    def __add__(self, molecule):
        return self.__class__(self.molecules + (molecule,))
    
    def serve_graphs(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=len(self),
            collate_fn=graph_collate,
        )
    
    
def graph_collate(batch):
    g = dgl.batch([molecule.graph for molecule in batch])
    h = g.ndata['h']
    y = torch.tensor([molecule.y for molecule in batch])
    return g, h, y

    
def assay(
    molecule: Molecule,
    past: Dataset,
    future: Dataset,
):
    molecule = future.lookup(molecule)
    future = future - molecule
    past = past + molecule
    return past, future

    
    



        
    