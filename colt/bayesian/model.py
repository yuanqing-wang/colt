import abc
import torch
from lightning import LightningModule

class WrappedModel(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        

        
    
    
    
    
        
        