import time
import matplotlib.pyplot as plt

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.policy_aggregator import PolicyAggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle, strategy_selectors
from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver

def get_aggregate_policy(game, PSRO):
    meta_probabilities = PSRO.get_meta_strategies()
    policies = PSRO.get_policies()

    return PolicyAggregator(game).aggregate(
        range(len(policies)), policies, meta_probabilities
    )

def get_exploitability(env, n_players, PSRO):
    policies = get_aggregate_policy(env.game, PSRO)
    return exploitability.nash_conv(env.game, policies)

def main():
    n_players = 3
    
    # Turn game into argument
    game = pyspiel.load_game_as_turn_based(
        "kuhn_poker", {"players": n_players}
    )
    env = rl_environment.Environment(game)

    random_policy = policy.TabularPolicy(env.game)
    oracle = best_response_oracle.BestResponseOracle(
        game=env.game, policy=random_policy
    )
    agents = [random_policy.__copy__() for _ in range(n_players)]

    N = 100

    PSRO = PSROSolver(
        env.game,
        oracle,
        initial_policies=agents,
        training_strategy_selector=strategy_selectors.probabilistic,
        sims_per_entry=N,
        sample_from_marginals=True,
        # Oracle kwargs
        symmetric_game=False,
    )
    
    log_file = open("data.dat", "w")
    for _ in range(10):
        for it in range(8):
            start_time = time.time()
            PSRO.iteration()
            end_time = time.time()
            elapsed = end_time - start_time

            meta_game = PSRO.get_meta_game()
            n_agents = meta_game[0].shape[0]
            print(f"Iteration : {it}")
            print(f"Time: {elapsed}")
            print(get_exploitability(env, n_players, PSRO))

            log_file.write(f"{n_agents} {elapsed}\n")

    log_file.close()

if __name__ == "__main__":
    main()
