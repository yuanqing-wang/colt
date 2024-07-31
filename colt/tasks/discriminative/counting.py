from rdkit import Chem
from .discriminative_task import DiscriminativeTask

class NumberOfAtoms(DiscriminativeTask):
    """The number of atoms."""
    implementation: lambda mol: mol.GetNumAtoms()
    
class NumberOfHeavyAtoms(DiscriminativeTask):
    """The number of heavy atoms."""
    implementation: lambda mol: mol.GetNumHeavyAtoms()
    
class NumberOfBonds(DiscriminativeTask):
    """The number of bonds."""
    implementation: lambda mol: mol.GetNumBonds()
    
class NumberOfRings(DiscriminativeTask):
    """The number of rings."""
    implementation: lambda mol: Chem.GetSSSR(mol)
    
class NumberOfRotatableBonds(DiscriminativeTask):
    """The number of rotatable bonds."""
    implementation: lambda mol: Chem.CalcNumRotatableBonds(mol)
    
class NumberOfHBD(DiscriminativeTask):
    """The number of hydrogen bond donors."""
    implementation: lambda mol: Chem.CalcNumHBD(mol)
    
class NumberOfHBA(DiscriminativeTask):
    """The number of hydrogen bond acceptors."""
    implementation: lambda mol: Chem.CalcNumHBA(mol)
    
class NumberOfHeteroatoms(DiscriminativeTask):
    """The number of heteroatoms."""
    implementation: lambda mol: Chem.CalcNumHeteroatoms(mol)
    
class NumberOfAromaticRings(DiscriminativeTask):
    """The number of aromatic rings."""
    implementation: lambda mol: Chem.CalcNumAromaticRings(mol)
    

    