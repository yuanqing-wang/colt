import pandas as pd
import lightning as pl
from colt.active.bayesian.bnn import WrappedBNNModel
from colt.active.bayesian.acquisition import collate
from colt.active.molecule import Molecule, Dataset 
from colt.active.trial import Trial

def run():
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv")
    smiles = df["smiles"]
    observations = df["measured log solubility in mols per litre"]
    future = [Molecule(smiles=x, y=y) for x, y in zip(smiles, observations)]
    future = Dataset(future)
    future.featurize()
    data = future.serve_graphs()
    
    
    model = WrappedBNNModel(
        in_features=74,
        out_features=1,
        hidden_features=16,
    )
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data)
    
    
    
    
if __name__ == "__main__":
    run()
    
    