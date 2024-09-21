import pandas as pd
import lightning as pl
from colt.bayesian.gnn import GNN
from colt.bayesian.bnn import WrappedBNNModel
from colt.bayesian.gp import WrappedExactGPModel
from colt.bayesian.acquisition import BayesianAcquisition
from colt.llm import LLMAcquisition
from colt.acquisition import RandomAcquisition
from colt.molecule import Molecule, Dataset 
from colt.trial import Trial

def run(args):
    from dgllife.data import ESOL, FreeSolv, Lipophilicity
    data = locals()[args.data]()
    future = [Molecule(smiles=x, y=y.item()) for x, _, y in data]
    future = Dataset(future)
    
    if args.model == "bnn":
        future.featurize()
        gnn = GNN(in_features=74, hidden_features=16, out_features=2)
        model = WrappedBNNModel(representation=gnn)
        acquisition = BayesianAcquisition(model=model)
    elif args.model == "gp":
        future.featurize()
        gnn = GNN(in_features=74, hidden_features=16, out_features=16)
        model = WrappedExactGPModel(representation=gnn)
        acquisition = BayesianAcquisition(model=model)
    elif args.model == "llm":
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        acquisition = LLMAcquisition(model=model, tries=5)
    elif args.model == "random":
        acquisition = RandomAcquisition()
    trial = Trial(acquisition=acquisition, future=future, steps=30)
    trial.loop()
    trajectory = trial.trajectory.tolist()
    result = {"method": args.model, "data": args.data, "trajectory": trajectory}
    import json
    with open("result.json", "a") as f:
        json.dump(result, f)
        f.write("\n")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--model", type=str, default="gp")
    args = parser.parse_args()
    run(args)
    
    