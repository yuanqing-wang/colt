import pandas as pd
import lightning as pl
from colt.active.bayesian.bnn import WrappedBNNModel
from colt.active.bayesian.gp import WrappedExactGPModel
from colt.active.bayesian.acquisition import BayesianAcquisition
from colt.active.llm import LLMAcquisition
from colt.active.acquisition import RandomAcquisition
from colt.active.molecule import Molecule, Dataset 
from colt.active.trial import Trial
from colt.active.bayesian.gnn import GNN

def run(args):
    from dgllife.data import ESOL, FreeSolv, Lipophilicity
    data = locals()[args.data]()
    future = [Molecule(smiles=x, y=y.item()) for x, _, y in data]
    future = Dataset(future)
    
    if args.model == "bnn":
        future.featurize()
        gnn = GNN(in_features=74, hidden_features=16, out_features=2)
        model = WrappedBNNModel(representation=gnn)
        trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False)
        trainer.fit(model, future.serve_graphs())
        
    elif args.model == "gp":
        future.featurize()
        gnn = GNN(in_features=74, hidden_features=16, out_features=16)
        model = WrappedExactGPModel(representation=gnn)
        model._init(future.serve_graphs())
        trainer = pl.Trainer(max_epochs=1000, enable_checkpointing=False)
        trainer.fit(model, future.serve_graphs())
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--model", type=str, default="gp")
    args = parser.parse_args()
    run(args)
    
    