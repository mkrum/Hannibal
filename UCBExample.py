import random
import logging
import numpy as np
from itertools import product
from copy import copy

import numpy as np


class GaussianOnePopGame:

    def __init__(self, actions, noise):
        self.actions = actions
        self.noise = noise

        # randomly select means for each action in [0, 1]
        self.means = np.random.random(size=(actions, actions))

    def get_entry_sample(self, entry):
        mean_val = self.means[tuple(entry)]
        
        # Get the amount that player one wins by, clipped 
        player1_win = np.random.normal(mean_val, self.noise, size=1)
        player1_win = np.clip(
            player1_win, mean_val - self.noise, mean_val + self.noise
        )

        return np.array([player1_win])

    def true_payoffs(self):
        return self.means

    def get_size(self):
        return self.actions


class FreqBandit:

    def __init__(
        self,
        means,
        counts,
        max_payoff=1,
        min_payoff=-1,
        delta=0.01,
    ):
        
        self.num_players = 2

        self.means = means
        self.counts = counts

        self.num_strats = means.shape[0]

        self.delta = delta
        self.range = max_payoff - min_payoff

        shape = tuple(self.num_strats for _ in range(self.num_players))

        self.logger = logging.getLogger("Freq_Bandit")
        np.random.seed()

        self.unresolved_pairs = set()

        for base_strat in product(range(self.num_strats), repeat=self.num_players):
            for n in range(self.num_players):
                # For each player that can deviate
                for strat_index in range(self.num_strats):
                    # For each strategy they can change to
                    if strat_index == base_strat[n]:
                        continue  # Not a different strategy, move on
                    new_strat = copy(list(base_strat))
                    new_strat[n] = strat_index
                    new_strat = tuple(new_strat)

                    unresolved_pair = (base_strat, new_strat)
                    self.unresolved_pairs.add(unresolved_pair)

        self._update_unresolved()

    def is_resolved(self):
        return len(self.unresolved_pairs) == 0

    def choose_entry_to_sample(self):

        # Uniformly pick a an unresolved pair and uniformly pick a strategy
        # TODO: Non uniform sampling -- pick based on performance.
        pair_idx = np.random.choice(len(self.unresolved_pairs))
        pair = list(self.unresolved_pairs)[pair_idx]
        strat = pair[random.randint(0, 1)]

        return strat

    def update_entry(self, strats, payoff):
        # Add count and payoff
        self.counts[strats] += 1
        N = self.counts[strats]

        self.means[strats] = (
            (N - 1) * self.means[strats] + payoff
        ) / N

        self._update_unresolved()
    
    def _update_unresolved(self):
        # Update the unresolved strategy pairs
        # Brute force for now
        pairs_to_remove = set()
        for pair in self.unresolved_pairs:
            base_strat, new_strat = pair

            # Test if the confidence intervals don't overlap
            # multiplayer games (all of them have to be non-overlapping pairwise)
            # Test of differences between them.

            bm = self.means[tuple(base_strat)]
            bc = self.counts[tuple(base_strat)]

            nm = self.means[tuple(new_strat)]
            nc = self.counts[tuple(new_strat)]

            if bc == 0 or nc == 0:
                # We have no observed evaluations, CI overlaps
                continue  

            base_lower, base_upper = compute_interval(self.range, self.delta, bm, bc)
            new_lower, new_upper = compute_interval(self.range, self.delta, nm, nc)

            totally_above = (bm >= nm) and (base_lower > new_upper)
            totally_below = (bm < nm) and (base_upper < new_lower)

            # Remove the pair
            if totally_above or totally_below:
                pairs_to_remove.add(pair)

        for pair_to_remove in pairs_to_remove:
            self.unresolved_pairs.discard(pair_to_remove)

    def payoff_distrib(self):
        # The variance of the estimates are not accurate,
        # variance is calculated in the 'update_entry' method
        return self.means, np.zeros_like(self.means)

def compute_interval(range_, delta, mean, n):
    interval = np.sqrt((np.log(2 / delta) * (range_ ** 2)) / (2 * n))
    return (mean - interval, mean + interval)

def run_sampling(
    payoff_matrix_sampler, sampler, max_iters=100, 
):
    t = 0
    while t < max_iters and not sampler.is_resolved():
        # Pick an entry to sample
        entry_to_sample = sampler.choose_entry_to_sample()

        # Get a sample from that entry
        payoff_samples = payoff_matrix_sampler.get_entry_sample(entry_to_sample)

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)

        t += 1

    return {
        "payoff_matrix_mean": sampler.payoff_distrib()[0],
        "unresolved": sampler.unresolved_pairs,
        "counts": sampler.counts,
    }


if __name__ == "__main__":

    # Notes from Geoff on the structure:
    # I didn't change the structure of the original code too much

    # The code assumes non-symmetric games, which means that it may not be optimized
    # for the symmetric case (it definitely isnt)

    # run_sampling takes in the game, however the game is more of a
    # payoff matrix sampler than a game, you tell it what actions to sample
    # and the game outputs payoff.

    game = GaussianOnePopGame(4, 0.1)

    sampler = FreqBandit(
        game.get_size(),
        max_payoff=1,
        min_payoff=0,
        delta=0.1,
    )  

    out = run_sampling(game, sampler, max_iters=1000)
    print(out['counts'])
    print(f"Final estimated payoff matrix:\n{out['payoff_matrix_mean']}")
    # This will only work for the 2 agent case, but here's the variance
    print(
        f"Estimated variance:\n{np.sqrt((np.log(2/0.1) * ((1 - 0)**2))/(2*out['counts']))}"
    )
    print(f"Unresolved cases:\n{out['unresolved']}")
