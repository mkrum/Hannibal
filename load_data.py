
import ast
import matplotlib.pyplot as plt
import numpy as np

def load_data(path):
    with open(path, 'r') as f:
        raw_data = f.readlines()
    
    data = []
    for d in raw_data:
        dict_ = ast.literal_eval(d)
        dict_['meta_game'] = np.array(dict_['meta_game'])
        data.append(dict_)
    return data


labels = []
for prd in [50000]:
    for N in [1, 10, 100]:
        data = load_data(f'data_{prd}_{N}.dat')
        times = [d['total_elapse'] for d in data]
        for i in range(len(times) - 1):
            times[i+1] += times[i]

        e = [d['exploit'] for d in data]
        labels.append(f'{prd}, {N}')
        plt.plot(e)

plt.legend(labels)
plt.ylabel("Exploitability")
plt.xlabel("Time (s)")
plt.yscale('log')
plt.show()
