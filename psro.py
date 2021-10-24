import time
import tqdm
import pickle as pkl

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


def meta_game_to_list(meta_game):
    for i in range(len(meta_game)):
        meta_game[i] = meta_game[i].tolist()
    return meta_game


def debug_iteration(psro):
    psro._iterations += 1

    info = {
            "type": "step", 
           }
    
    # Run the Oracle and get the new agent
    first_start = time.time()
    psro.update_agents()
    end = time.time()

    info["oracle_elapse"] = end - first_start

    # Update the payoff matrix
    start = time.time()
    psro.update_empirical_gamestate()
    end = time.time()

    info["sims_elapse"] = end - start

    # Run the meta-solver
    start = time.time()
    psro.update_meta_strategies()
    last_end = time.time()

    info["meta_solve_elapse"] = last_end - start

    info["total_elapse"] = (
        info["oracle_elapse"] + info["sims_elapse"] + info["meta_solve_elapse"]
    )

    info["meta_game"] = pkl.dumps(psro.get_meta_game())
    info["meta_strategies"] = pkl.dumps(psro.get_meta_strategies())
    info["policies"] = pkl.dumps(psro.get_policies())
    return info


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
    n_players = 2

    # Turn game into argument
    game = pyspiel.load_game_as_turn_based("kuhn_poker", {"players": n_players})
    env = rl_environment.Environment(game)

    random_policy = policy.TabularPolicy(env.game)

    oracle = best_response_oracle.BestResponseOracle(
        game=env.game, policy=random_policy
    )

    agents = [random_policy.__copy__() for _ in range(n_players)]

    runs = 1
    steps = 100
    meta_strategy = "prd"

    for prd in [10000, 30000, 50000]:
        for N in [1, 10, 100]:

            log_file = open(f"debug_data_{prd}_{N}.dat", "w")

            for _ in range(runs):
                PSRO = PSROSolver(
                    env.game,
                    oracle,
                    meta_strategy_method="prd",
                    initial_policies=agents,
                    training_strategy_selector=strategy_selectors.probabilistic,
                    sims_per_entry=N,
                    sample_from_marginals=True,
                    # Oracle kwargs
                    symmetric_game=False,
                    # PRD kwargs
                    prd_iterations=prd,
                    prd_gamma=1e-10,
                )
                exploit = get_exploitability(env, n_players, PSRO)
                info = {
                    "type": "init",
                    "meta_strategy": meta_strategy,
                    "meta_game": pkl.dumps(PSRO.get_meta_game()),
                    "meta_strategies": pkl.dumps(PSRO.get_meta_strategies()),
                    "policies": pkl.dumps(PSRO.get_policies()),
                    "prd_iterations": prd,
                    "prd_gamma": 1e-10,
                    "starting_exploit": exploit,
                }
                log_file.write(str(info) + "\n")
                print("intial exploit ", str(exploit))

                for it in tqdm.tqdm(range(steps)):
                    info = debug_iteration(PSRO)
                    info["it"] = it

                    exploit = get_exploitability(env, n_players, PSRO)
                    info["exploit"] = exploit

                    log_file.write(str(info) + "\n")
                    log_file.flush()

    log_file.close()


if __name__ == "__main__":
    main()
