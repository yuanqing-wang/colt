import numpy as np
from matplotlib import pyplot as plt

# use heltiva as the default font
plt.rcParams['font.family'] = 'serif'

# change default line color 
import matplotlib as mpl
import numpy as np

def run():
    import json
    results = open('result.json', 'r').readlines()
    results = [json.loads(r) for r in results]
    
    DATA = "FreeSolv"
    results_llm = [r for r in results if r['method'] == 'llm' and r['data'] == DATA]
    results_bnn = [r for r in results if r['method'] == 'bnn' and r['data'] == DATA]
    results_random = [r for r in results if r['method'] == 'random' and r['data'] == DATA]
    results_gp = [r for r in results if r['method'] == 'gp' and r['data'] == DATA]
    
    traj_llm = [r['trajectory'] for r in results_llm]
    traj_bnn = [r['trajectory'] for r in results_bnn]
    traj_random = [r['trajectory'] for r in results_random]
    traj_gp = [r['trajectory'] for r in results_gp]
    traj_llm = np.array(traj_llm)
    traj_bnn = np.array(traj_bnn)
    traj_random = np.array(traj_random)
    traj_gp = np.array(traj_gp)
    traj_llm = np.maximum.accumulate(traj_llm, axis=1)
    traj_bnn = np.maximum.accumulate(traj_bnn, axis=1)
    traj_random = np.maximum.accumulate(traj_random, axis=1)
    traj_gp = np.maximum.accumulate(traj_gp, axis=1)
    
    plt.plot(traj_llm.mean(axis=0), label='llm')
    plt.plot(traj_bnn.mean(axis=0), label='bnn')
    plt.plot(traj_random.mean(axis=0), label='random')
    plt.plot(traj_gp.mean(axis=0), label='gp')
    plt.legend()
    
    plt.savefig('plot.png') 
    
if __name__ == '__main__':
    run()