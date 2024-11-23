import pandas as pd
from .utils import scale

def mpro():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/meyresearch/ActiveLearning_BindingAffinity/refs/heads/main/Mpro_final.csv",
    )
    x = df["SMILES"]
    y = df["affinity"]
    y = scale(y)
    return x, y

def tyk2():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/meyresearch/ActiveLearning_BindingAffinity/refs/heads/main/TYK2_final.csv",
    )
    x = df["SMILES"]
    y = df["affinity"]
    y = scale(y)
    return x, y