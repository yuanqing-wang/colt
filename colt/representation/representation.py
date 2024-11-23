import torch
import abc
from ..molecule import Dataset

class Representation(torch.nn.Module):
    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__()

    