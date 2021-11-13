
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


def compute_meta_game(psro, sampler, N, history):
    """Given new agents in _new_policies, update meta_game through simulations.

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    budget = N * (2 * len(psro._policies[0]) + 1)
    og_len = len(history)
    meta_game = sampler(psro, budget, history)
    assert (len(history) - og_len) <= budget

    meta_game = [meta_game, -meta_game]

    psro._meta_games = meta_game
    psro._policies = [psro._policies[k] + psro._new_policies[k] for k in range(psro._num_players)]
