
import ast
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import glob

sns.set()

def load_data(path):
    with open(path, 'r') as f:
        raw_data = f.readlines()
    
    data = []
    for d in raw_data:
        dict_ = ast.literal_eval(d)

        if 'meta_game' in dict_.keys():
            dict_['meta_game'] = pkl.loads(dict_['meta_game'])
        if 'meta_strategies' in dict_.keys():
            dict_['meta_strategies'] = pkl.loads(dict_['meta_strategies'])

        data.append(dict_)
    return data

def is_done(lines, path):
    with open(path, 'r') as f:
        return len(f.readlines()) >= lines

def get_exploit(path):
    data = load_data(path)
    exploits = np.zeros(100)

    data = list(filter(lambda x: 'exploit' in x.keys(), data))
    for (i, d) in enumerate(data):
        exploits[i] += float(d['exploit'])

    return exploits

cutoff = 50
for g in ['fixed/baseline_uniform_1_50000*', 'fixed/baseline_uniform_5_50000*', 'fixed/baseline_uniform_10_50000*','fixed/baseline_uniform_100_50000*', 'test/*']:
    files = glob.glob(g)
    print(len(files))
    files = list(filter(lambda x: is_done(cutoff + 1, x), files))

    total_exp = np.zeros((len(files), cutoff))

    for (i, f) in enumerate(files):
        total_exp[i] = get_exploit(f)[:cutoff]

    mean_vals = np.mean(total_exp, axis=0)
    max_vals = np.percentile(total_exp, 95, axis=0)
    min_vals = np.percentile(total_exp, 5, axis=0)
    
    plt.yscale("log")
    plt.plot(mean_vals)
    #plt.fill_between(range(len(min_vals)), min_vals, max_vals, alpha=0.5)

plt.legend(["1", "5", "10", "100", "UCB"])
plt.show()
