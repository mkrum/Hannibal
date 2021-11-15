
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

    return var_mat

def sample(psro, history, i, j):
    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    samp = [policies[0][i], policies[1][j]]
    out = psro.sample_episodes(samp, 1)[0]
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

def baseline_uniform_binary(psro, budget, history):
    out = baseline_uniform(psro, budget, history)
    return np.sign(out)

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
        max_payoff=1,
        min_payoff=-1,
        delta=0.1,
    )  

    t = 0
    while t < budget and not sampler.is_resolved():
        # Pick an entry to sample
        entry_to_sample = sampler.choose_entry_to_sample()
        
        # Get a sample from that entry
        payoff_samples = sample(psro, history, *entry_to_sample)

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)

        t += 1

    meta_game = compute_matrix(len(policies[0]), history)
    return meta_game


def compute_bounds(mean, counts, delta):
    range_ = 4
    interval = np.sqrt((np.log(2 / delta) * (range_ ** 2)) / (2 * counts))
    return (mean - interval, mean + interval)

def simple_ucb(psro, budget, history):

    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    
    for _ in range(5):
        size = len(policies[0]) - 1
        for i in range(size):
            sample(psro, history, i, size)
            sample(psro, history, size, i)
            budget -= 2

        sample(psro, history, size, size)
        budget -= 1

    means = compute_matrix(len(policies[0]), history)
    counts = get_counts(len(policies[0]), history)

    d = 0.01

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

def gap(psro, budget, history):

    policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
    
    for _ in range(5):
        size = len(policies[0]) - 1
        for i in range(size):
            sample(psro, history, i, size)
            sample(psro, history, size, i)
            budget -= 2

        sample(psro, history, size, size)
        budget -= 1
    
    counts = get_counts(len(policies[0]), history)
    var = compute_variance(len(policies[0]), history)
    #print(var.shape)

    mask = np.zeros_like(var)
    mask[-1, :] = 1
    mask[:, -1] = 1
    
    gap = mask * (var / counts)
    while budget > 0:
        biggest_gap = np.unravel_index(np.argmax(gap, axis=None), gap.shape)
        i, j = biggest_gap
        out = sample(psro, history, i, j)

        var = compute_variance(len(policies[0]), history)
        counts[i, j] += 1
        gap = mask * (var / counts)

        budget -= 1
    
    #print(counts)
    counts = get_counts(len(policies[0]), history)
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
    meta_game = sampler(psro, budget - len(history), history)
    assert len(history) <= budget

    meta_game = [meta_game, -meta_game]

    psro._meta_games = meta_game
    psro._policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
