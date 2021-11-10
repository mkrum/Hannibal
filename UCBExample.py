### MVP UCB

import numpy as np
import logging
import numpy as np
import logging
from itertools import product
from copy import copy
import random

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.policy_aggregator import PolicyAggregator
from open_spiel.python.algorithms.psro_v2 import (
    best_response_oracle,
    strategy_selectors,
)
from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver
from open_spiel.python.algorithms.psro_v2.optimization_oracle import AbstractOracle

class GaussianOnePopGames:
    
    def __init__(self, actions=2, seed=None, noise=1, clip=True):
        self.logger = logging.getLogger("GaussianGamesOnePop")
        self.actions = actions
        self.noise = noise
        self.clip = clip
        if seed is not None:
            np.random.seed(seed)
        self.matrix = np.random.random(size=(1, actions, actions))
        self.logger.debug("\n"+str(np.around(self.matrix, 2)))
        if seed is not None:
            np.random.seed()

    def get_entry_sample(self, entry):
        player1_win = np.random.normal(self.matrix[0][tuple(entry)], self.noise, size=1)
        if self.clip:
            mean_val = self.matrix[0][tuple(entry)]
            player1_win = np.clip(player1_win, mean_val - self.noise, mean_val + self.noise)
        return np.array([player1_win])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions

class FreqBandit:
    
    def __init__(self, num_pops, num_strats, num_players, max_payoff=1, min_payoff=0, delta=0.1, alpha_rank_func=None):
        self.num_pops = num_pops
        self.num_strats = num_strats
        self.num_players = num_players

        self.delta = delta
        self.range = max_payoff - min_payoff

        shape = (num_pops, *[num_strats for _ in range(num_players)])
        self.means = np.zeros(shape=shape)
        self.counts = np.zeros(shape=shape)

        self.logger = logging.getLogger("Freq_Bandit")
        np.random.seed()

        self.unresolved_pairs = set()
        if self.num_pops == 1:
            for i in range(num_strats):
                for j in range(num_strats):
                    if i == j:
                        continue
                    self.unresolved_pairs.add((
                        (i,j),
                        (j,i),
                        0
                    ))
        else:
            for base_strat in product(range(num_strats), repeat=num_players):
                for n in range(num_players):
                    # For each player that can deviate
                    for strat_index in range(num_strats):
                        # For each strategy they can change to 
                        if strat_index == base_strat[n]:
                            continue # Not a different strategy, move on
                        new_strat = copy(list(base_strat))
                        new_strat[n] = strat_index
                        new_strat = tuple(new_strat)
                        for p in range(num_pops):
                            unresolved_pair = (base_strat, new_strat, p)
                            self.unresolved_pairs.add(unresolved_pair)

    def choose_entry_to_sample(self):
        if len(self.unresolved_pairs) == 0:
            return None, {}
        self.logger.debug("Unresolved pairs has {} elements".format(len(self.unresolved_pairs)))

        # Uniformly pick a an unresolved pair and uniformly pick a strategy
        # TODO: Non uniform sampling -- pick based on performance.
        pair = random.sample(self.unresolved_pairs, k=1)[0]
        strat = pair[random.randint(0, 1)]
        return strat, {}

    def update_entry(self, strats, payoffs):
        # Add count and payoff
        for player, payoff in enumerate(payoffs):
            self.counts[player][tuple(strats)] += 1
            N = self.counts[player][tuple(strats)]
            self.means[player][tuple(strats)] = ((N - 1) * self.means[player][tuple(strats)] + payoff) / N

        # Update the unresolved strategy pairs
        # Brute force for now
        pairs_to_remove = set()
        for pair in self.unresolved_pairs:
            base_strat, new_strat, p = pair
            
            # Test if the confidence intervals don't overlap
            # multiplayer games (all of them have to be non-overlapping pairwise)
            # Test of differences between them.
            bm = self.means[p][tuple(base_strat)]
            bc = self.counts[p][tuple(base_strat)]
            nm = self.means[p][tuple(new_strat)]
            nc = self.counts[p][tuple(new_strat)]
            if bc == 0 or nc == 0:
                continue # We have no observed evaluations, CI overlaps
            
            bi = np.sqrt((np.log(2/self.delta) * (self.range**2))/(2*bc))
            base_lower = bm - bi
            base_upper = bm + bi
            
            ni = np.sqrt((np.log(2/self.delta) * (self.range**2))/(2*nc))
            new_lower = nm - ni
            new_upper = nm + ni
            

            if bm >= nm and new_upper > base_lower:
                pass
            elif bm < nm and base_upper > new_lower:
                pass
            else:
                # Remove the pair
                pairs_to_remove.add(pair)
        
        for pair_to_remove in pairs_to_remove:
            self.unresolved_pairs.discard(pair_to_remove)

    def payoff_distrib(self):
        # The variance of the estimates are not accurate,
        # variance is calculated in the 'update_entry' method
        return np.copy(self.means), np.zeros_like(self.means)
    
def run_sampling(payoff_matrix_sampler, sampler, max_iters=100, graph_samples=10, true_payoff=None):
    logger = logging.getLogger("Sampling")
    logger.warning("Starting sampling for up to {:,} iterations.".format(max_iters))
    

    improvements = []
    entries = []
    payoff_matrix_means = []
    payoff_matrix_vars = []

    for t in range(max_iters):
        if True:
            m_means, m_vars = sampler.payoff_distrib()
            payoff_matrix_means.append(m_means)
            payoff_matrix_vars.append(m_vars)
        
        # Pick an entry to sample
        entry_to_sample, sampler_stats = sampler.choose_entry_to_sample()
        if entry_to_sample is None:
            break
        entries.append(entry_to_sample)
        logger.info("Sampling {}".format(entry_to_sample))
        
        # Get a sample from that entry
        payoff_samples = payoff_matrix_sampler.get_entry_sample(entry_to_sample)
        logger.info("Received Payoff {} for {}".format(payoff_samples, entry_to_sample))

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)

    logger.critical("Finished {} iterations".format(t))
    unresolved = sampler.unresolved_pairs

    counts = sampler.counts
    del sampler

    return {
        "entries": entries,
        "payoff_matrix_means": payoff_matrix_means,
        "payoff_matrix_vars": payoff_matrix_vars,
        "unresolved": unresolved,
        'counts': counts
    }


if __name__ == '__main__':
    
    # Notes from Geoff on the structure:
    # I didn't change the structure of the original code too much
    
    # The code assumes non-symmetric games, which means that it may not be optimized
    # for the symmetric case (it definitely isnt)
    
    # run_sampling takes in the game, however the game is more of a
    # payoff matrix sampler than a game, you tell it what actions to sample
    # and the game outputs payoff.
    
    delta = 0.1
    max_payoff = 1
    min_payoff = 0
    game = GaussianOnePopGames(actions=4, seed=3)
    num_pops, num_players, num_strats = game.get_env_info()
    sampler = FreqBandit(num_pops, 
                       num_strats, 
                       num_players, 
                       max_payoff = 1,
                       min_payoff = 0,
                       delta=0.2, 
                       alpha_rank_func=None) # Don't need alpharank function, I removed it
    
    out = run_sampling(game, sampler, max_iters=500)
    print(f"Number of iterations until agreement: {out['counts'].sum()}")
    print(f"Final estimated payoff matrix:\n{out['payoff_matrix_means'][-1]}")
    # This will only work for the 2 agent case, but here's the variance
    print(f"Estimated variance:\n{np.sqrt((np.log(2/0.1) * ((max_payoff - min_payoff)**2))/(2*out['counts']))}")
    print(f"Unresolved cases:\n{out['unresolved']}")
