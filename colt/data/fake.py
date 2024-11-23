from rdkit import Chem
import pandas as pd
import torch

def fake_target(smiles, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganGenerator().GetFingerprint(mol)
    fp = torch.tensor(fp, dtype=torch.float32).flatten()
    with torch.random.fork_rng():
        torch.random.manual_seed(seed)
        projection = torch.randn(len(fp), 1)
    return (projection * fp).sum()

def fake_targets(smiles, seed):
    out = torch.tensor([fake_target(s, seed) for s in smiles])
    min, max = out.min(), out.max()
    out = (out - min) / (max - min)
    return out

def fake_esol(seed):
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv')
    x = df['smiles']
    y = fake_targets(x, seed)
    return x, y

def fake_freesolv(seed):
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv')
    x = df['smiles']
    y = fake_targets(x, seed)
    return x, y

def fake_lipophilicity(seed):
    df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv')
    x = df['smiles']
    y = fake_targets(x, seed)
    return x, y

def fake_zinc(seed=0, size=100):
    df = pd.read_csv("https://github.com/aspuru-guzik-group/chemical_vae/raw/refs/heads/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv")
    df = df.sample(size, random_state=seed)
    x = df['smiles']
    y = fake_targets(x, seed)
    return x, y

    