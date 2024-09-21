import torch
import pyro

import functools
from .gnn import GNN
from .model import WrappedModel
from pyro.nn.module import to_pyro_module_
NUM_SAMPLES = 8

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def init_sigma(model, value):
    """Initializes the log_sigma parameters of a model

    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize

    value : float
        The value to initialize the log_sigma parameters to

    """
    params = {}

    for name, base_name in model.buffered_params.items():
        mean_name = base_name + "_mean"
        sigma_name = base_name + "_sigma"
        mean = getattr(model, mean_name)
        sigma = getattr(model, sigma_name)
        if mean.dim() == 1:
            continue
        params[name] = pyro.nn.PyroSample(
            pyro.distributions.Normal(
                mean,sigma
            ).to_event(mean.dim())
        )

    for name, param in params.items():
        rsetattr(model, name, params[name])

class UnwrappedBNNModel(torch.nn.Module):
    def __init__(
        self,
        representation: torch.nn.Module,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.representation = representation
        self._prepare_buffer()
    
    def _prepare_buffer(self):
        buffered_params = {}
        for name, param in self.named_parameters():
            base_name = name.replace(".", "-")
            mean_name = base_name + "_mean"
            sigma_name = base_name + "_sigma"
            self.register_buffer(mean_name, torch.zeros(param.shape))
            self.register_buffer(sigma_name, torch.ones(param.shape) * self.sigma)
            buffered_params[name] = base_name
        self.buffered_params = buffered_params
        
    def forward(self, g, h, y=None):
        y_hat = self.representation(g, h)
        y_hat_mu, y_hat_log_sigma = y_hat.split(y_hat.shape[-1] // 2, dim=-1)
        with pyro.plate("data", g.batch_size):
            y_hat = pyro.sample(
                "obs", 
                pyro.distributions.Normal(
                    y_hat_mu.squeeze(-1), 
                    torch.nn.functional.softplus(y_hat_log_sigma).squeeze(-1),
                ), 
            obs=y)
                        
        return y_hat
    
class WrappedBNNModel(WrappedModel):
    _refresh = False
    def __init__(
        self, 
        representation: torch.nn.Module,
        autoguide: pyro.infer.autoguide.guides.AutoGuide = pyro.infer.autoguide.AutoDiagonalNormal,
        sigma: float = 1.0,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss: torch.nn.Module = pyro.infer.Trace_ELBO(
            num_particles=NUM_SAMPLES,
            vectorize_particles=True,
        ),
    ):
        
        model = UnwrappedBNNModel(representation=representation)
        to_pyro_module_(model)
        init_sigma(model, sigma)
        super().__init__(model=model)

        self.guide = autoguide(self.model)

        # initialize head
        self.automatic_optimization = False
        self.save_hyperparameters()

        # initialize optimizer
        optimizer = getattr(pyro.optim, optimizer)(
            {"lr": lr, "weight_decay": weight_decay},
        )

        self.svi = pyro.infer.SVI(
            self.model.forward,
            self.guide,
            optim=optimizer,
            loss=loss,
            num_samples=NUM_SAMPLES,
        )

    def configure_optimizers(self):
        return None

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.model.cuda(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().cuda(*args, **kwargs)
    
    def cpu(self, *args, **kwargs):
        self.model.cpu(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().cpu(*args, **kwargs)
    
    def forward(self, g, h):
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=True,
            return_sites=["_RETURN", "obs"],
        )

        y_hat = predictive(g, h)["obs"]
        return y_hat
    
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        g, h, y = batch
        h = g.ndata["h"]
        y = y.float()
        loss = self.svi.step(g, h, y)


        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    