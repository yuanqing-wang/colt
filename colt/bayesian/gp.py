from typing import Optional
from ..molecule import Dataset
import torch
import dgl
import gpytorch
from .model import WrappedModel
NUM_SAMPLES = 8
# gpytorch.settings.debug._state = False

class UnwrappedExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, inputs, targets, representation, graph):
        super().__init__(inputs, targets, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.LinearMean(representation.hidden_features)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.representation = representation
        self.graph = graph

    def forward(self, x):
        x = self.representation(self.graph, x)
        mean = self.mean_module(x.tanh())
        covar = self.covar_module(x.tanh())
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
class WrappedExactGPModel(WrappedModel):
    def __init__(
        self,
        representation: torch.nn.Module,
        data: Optional[torch.utils.data.DataLoader] = None,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        
        super().__init__(model=None)
        self.representation = representation
        self.save_hyperparameters()
            
    def _init(self, data):
        g, h, y = next(iter(data))
        self.model = UnwrappedExactGPModel(
            inputs=h, 
            targets=y, 
            graph=g,
            representation=self.representation,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    
    def forward(self, g, h):
        self.model.graph = g
        y_hat = self.model.forward(h).sample(sample_shape=torch.Size((NUM_SAMPLES,)))
        return y_hat
    
    def training_step(self, batch, batch_idx):
        g, h, y = batch
        self.model.graph = g
        y_hat = self.model.forward(h)
        nll = -self.mll(y_hat, y).mean()
        return nll
        
    def configure_optimizers(self):
        # initialize optimizer
        optimizer = getattr(
            torch.optim,
            self.hparams.optimizer,
        )(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


        
        
        
        
        
        
        
        