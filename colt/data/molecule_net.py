import pandas as pd
from .utils import scale

def esol():
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv')
    x = df['smiles']
    y = df['measured log solubility in mols per litre'].values
    y = scale(y)
    return x, y

def freesolv():
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv')
    x = df['smiles']
    y = df['expt'].values
    y = scale(y)
    return x, y

def lipophilicity():
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv')
    x = df['smiles']
    y = df['exp'].values
    y = scale(y)
    return x, y