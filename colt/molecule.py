from dataclasses import dataclass
import abc
from typing import Optional, Any
from functools import lru_cache
import torch
import numpy as np
import dgl

@dataclass
class Molecule:
    smiles: str
    y: Optional[float] = None
    feature: Optional[Any] = None
    
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
    
    def featurize(self, featurizer):
        self.feature = featurizer(self)
    
class Featurizer:
    def __call__(self, molecule: Molecule):
        raise NotImplementedError
    
class GraphFeaturizer(Featurizer):
    def __call__(self, molecule: Molecule):
        from dgllife.utils import (
            smiles_to_bigraph,
            CanonicalBondFeaturizer,
            CanonicalAtomFeaturizer,
        )
                
        return smiles_to_bigraph(
            molecule.smiles,
            node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'),
            edge_featurizer=None,
        )
        
class FingerprintFeaturizer(Featurizer):
    @lru_cache(maxsize=None)
    def __call__(self, molecule):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        smiles = molecule.smiles
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganGenerator().GetFingerprint(mol)
        return torch.tensor(fp, dtype=torch.float32).flatten()
    
class LLMFeaturizer(Featurizer):
    def __init__(self, model, header="", observation=""):
        from transformers import pipeline
        self.model = pipeline("feature-extraction", model=model, device=0, trust_remote_code=True)
        self.header = header
        if observation:
            observation = observation + " of "
        self.observation = observation
        
    def __call__(self, molecule):
        fp = self.model(self.header + self.observation + molecule.smiles)
        fp = torch.tensor(fp, dtype=torch.float32).mean(0).mean(0)
        return fp
    
class RawLLMFeaturizer(Featurizer):
    def __init__(self, model, header="", observation=""):
        from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(0)
        self.header = header
        
    def __call__(self, molecule):
        tokenized = self.tokenizer(self.header + molecule.smiles, return_tensors="pt").to(0)
        fp = self.model(**tokenized, decoder_input_ids=tokenized.input_ids).last_hidden_state.mean(0).mean(0)
        return fp

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
    
    def shuffle(self):
        self.molecules = tuple(np.random.permutation(self.molecules))
        
    def featurize(self, featurizer):
        for molecule in self.molecules:
            molecule.feature = featurizer(molecule)
            
    def batch(self):
        y = torch.tensor([molecule.y for molecule in self.molecules], dtype=torch.float32)
        if isinstance(self.molecules[0].feature, dgl.DGLGraph):
            x = dgl.batch([molecule.feature for molecule in self.molecules])
        else:
            x = torch.stack([molecule.feature for molecule in self.molecules])
        return x, y
    
def assay(
    molecule: Molecule,
    past: Dataset,
    future: Dataset,
):
    molecule = future.lookup(molecule)
    future = future - molecule
    past = past + molecule
    return past, future

    
    



        
    