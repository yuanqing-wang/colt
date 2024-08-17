import pandas as pd
from colt.active.llm import LLMAcquisition
from colt.active.molecule import Molecule, Dataset 
from colt.active.trial import Trial

def run():
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv")
    smiles = df["smiles"]
    observations = df["measured log solubility in mols per litre"]
    future = [Molecule(smiles=x, y=y) for x, y in zip(smiles, observations)]
    future = Dataset(future)
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    acquisition = LLMAcquisition(model=model, tries=1)
    trial = Trial(future=future, acquisition=acquisition, steps=10)
    trial.loop()
    print(trial.past)
    
    
    
    
if __name__ == "__main__":
    run()
    
    