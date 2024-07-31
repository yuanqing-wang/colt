import abc
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class Task:
    """Base class for tasks. """
    description: str
    implementation: Callable
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    
    
    
    