from dataclasses import dataclass
from typing import Optional, Callable
from rdkit import Chem


def _if_str_then_mol(fn):
    """Function decorator that converts a string to a molecule for inputs. """
    def _wrapper(x):
        if isinstance(x, str):
            x = Chem.MolFromSmiles(x)
        return fn(x)
    return _wrapper

@dataclass
class Task:
    """Base class for tasks. """
    implementation: Callable
    description: Optional[str] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        # If the name is not provided, use the class name, separated by spaces.
        if self.name is None:
            name = self.__class__.__name__
            
            # Split the name by capital letters.
            import re
            name = re.findall(r'[A-Z][^A-Z]*', name)
            name = " ".join(name)
            self.name = name
            
        # wrap the implementation in the _if_str_then_mol decorator.
        self.implementation = _if_str_then_mol(self.implementation)
        
        # if there is no description, parse the docstring.
        if self.description is None:
            self.description = self.implementation.__doc__.lower()
        
            
            
            
            

    
    
    
    