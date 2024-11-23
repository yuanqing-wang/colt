import numpy as np

def test_single_round():
    import pandas as pd
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv")
    x = df["smiles"]
    y = df["measured log solubility in mols per litre"]
    
    train_idxs = np.random.choice(range(len(x)), size=10, replace=False)
    test_idxs = np.random.choice(range(len(x)), size=10, replace=False)
    
    portfolio = list(zip(x[train_idxs], y[train_idxs]))
    pool = list(x[test_idxs])
    
    from colt.active.llm import pick
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    for _ in range(5): # 5 tries
        result = pick(portfolio, pool, model)
        if result in pool:
            break
    print(result)
    
if __name__ == "__main__":
    test_single_round()