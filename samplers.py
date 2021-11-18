
import itertools
import numpy as np
from UCBExample import FreqBandit

def get_counts(N, history):
    counts = np.zeros((N, N))

    for ((i, j), _) in history:
        counts[i, j] += 1
    return counts

def compute_matrix(N, history):
    counts = get_counts(N, history)

    meta = np.zeros((N, N))
    for ((i, j), v) in history:
        meta[i, j] += v

    return meta / counts

def compute_variance(N, history):                                  
                                                                   
    possible = list(itertools.product(range(N), range(N)))         
                                                                   
    var = {p: [] for p in possible}                
                                                                   
    for ((i, j), v) in history:                                 
        var[i, j].append(v)                                        
                                                                   
    var_mat = np.zeros((N, N))
                                                                   
    for p in possible:                          
        var_mat[p] = np.var(var[p])
    
    counts = get_counts(N, history)

    return var_mat / counts


def sample(psro, history, i, j):
    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    samp = [policies[0][i], policies[1][j]]
    out = psro.sample_episodes(samp, 1)[0] + np.random.normal(0,2)
    history.append(((i, j), out))
    return out

def initialize_history(psro, N):
    history = []
    for _ in range(N):
        sample(psro, history, 0, 0)
    return history

def baseline_uniform(psro, budget, history):
    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    meta_game = compute_matrix(len(policies[0]), history)

    size = len(policies[0]) - 1

    for _ in range(budget // ((2 * size) + 1)):
        for i in range(size):
            sample(psro, history, i, size)
            sample(psro, history, size, i)

        sample(psro, history, size, size)
    
    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game
  
def random_sampler(psro, budget, history):
    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    meta_game = compute_matrix(len(policies[0]), history)
    
    size = len(policies[0]) - 1
    for i in range(size):
        sample(psro, history, i, size)
        sample(psro, history, size, i)
        budget -= 2

    sample(psro, history, size, size)
    budget -= 1
    
    policy_index = []
    for i in range(size):
      policy_index.append((size,i))
      policy_index.append((i,size))
    policy_index.append((size,size))
        
    
    while budget > 0:
      ind = np.random.choice(len(policy_index))
      sample(psro, history, policy_index[ind][0], policy_index[ind][1])
      budget -=1
    
    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game

def variance_sampler(psro, budget, history):
    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    meta_game = compute_matrix(len(policies[0]), history)
    
    size = len(policies[0]) - 1
    
    policy_index = []
    for i in range(size):
      policy_index.append((size,i))
      policy_index.append((i,size))
    policy_index.append((size,size))
    var = [[] for p in range(len(policy_index))]    
    for _ in range(3):
        for i, index in enumerate(policy_index):
            var[i].append(sample(psro, history, index[0], index[1]))
            budget -=1
    
    
    while budget > 0:
      avg_var = np.mean([(np.var(y))/(len(y)) for y in var])
      mean = [np.mean(y) for y in var]
      ind = np.argmax([ np.abs((np.var(y))/(len(y))) for y in var])
      var[ind].append(sample(psro, history, policy_index[ind][0], policy_index[ind][1]))
      budget -=1
    
    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game



def ucb(psro, budget, history):

    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    size = len(policies[0]) - 1
    for i in range(size):
        sample(psro, history, i, size)
        sample(psro, history, size, i)
        budget -= 2

    sample(psro, history, size, size)
    budget -= 1

    meta_game = compute_matrix(len(policies[0]), history)
    counts = get_counts(len(policies[0]), history)

    sampler = FreqBandit(
        meta_game,
        counts,
        max_payoff=2,
        min_payoff=-2,
        delta=1,
    )  

    t = 0
    while t < budget and not sampler.is_resolved():
        # Pick an entry to sample
        entry_to_sample = sampler.choose_entry_to_sample()
        
        # Get a sample from that entry
        payoff_samples = sample(psro, history, *entry_to_sample)

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)
        print(len(sampler.unresolved_pairs))
        t += 1

    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game


def compute_bounds(mean, counts, delta):
    range_ = 4
    interval = np.sqrt((np.log(2 / delta) * (range_ ** 2) / (2 * counts)))
    return (mean - interval, mean + interval)

def simple_ucb(psro, budget, history):

    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]

    size = len(policies[0]) - 1
    for i in range(size):
        sample(psro, history, i, size)
        sample(psro, history, size, i)
        budget -= 2

    sample(psro, history, size, size)
    budget -= 1

    means = compute_matrix(len(policies[0]), history)
    counts = get_counts(len(policies[0]), history)

    d = 1

    lower, upper = compute_bounds(means, counts, d)
    unresolved = (lower < 0.0) & (upper > 0.0)
    unresolved = np.nonzero(unresolved)

    while budget > 0 and len(unresolved[0]) > 0:
        sampled = np.random.choice(len(unresolved[0]))
        i = unresolved[0][sampled]
        j = unresolved[1][sampled]
        out = sample(psro, history, i, j)

        counts[i, j] += 1
        means[i, j] = ((counts[i, j] - 1) * means[i, j] + out) / counts[i, j]

        lower, upper = compute_bounds(means, counts, d)
        unresolved = (lower < 0.0) & (upper > 0.0)
        unresolved = np.nonzero(unresolved)

        budget -= 1
    
    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game

def compute_meta_game(psro, sampler, N, history):
    """Given new agents in _new_policies, update meta_game through simulations.

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    budget = sum(N * (2 * (i + 1) + 1) for i in range(-1, len(psro._policies[0])))
    print(f"budget={budget}")
    meta_game = sampler(psro, budget - len(history), history)
    assert len(history) <= budget

    meta_game = [meta_game, -meta_game]

    psro._meta_games = meta_game
    psro._policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
