
import time
import tqdm
import pickle as pkl
import string
import random
import argparse

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.policy_aggregator import PolicyAggregator
from open_spiel.python.algorithms.psro_v2 import ( best_response_oracle,
    strategy_selectors,
)
from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver
from open_spiel.python.algorithms.psro_v2.optimization_oracle import AbstractOracle

from samplers import *


def meta_game_to_list(meta_game):
    for i in range(len(meta_game)):
        meta_game[i] = meta_game[i].tolist()
    return meta_game


def debug_iteration(psro, sampler, N, history, verbose=False):
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
    compute_meta_game(psro, sampler, N, history)
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
    
    if verbose:
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

def generate_data(prd, N, data_dir, sampler, steps=100):
    n_players = 2

    game = pyspiel.load_game_as_turn_based("kuhn_poker", {"players": n_players})

    env = rl_environment.Environment(game)

    random_policy = policy.TabularPolicy(env.game)

    oracle = best_response_oracle.BestResponseOracle(
        game=env.game, policy=random_policy
    )

    agents = [random_policy.__copy__() for _ in range(n_players)]
        
    random_extension = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    print(random_extension)
    log_file = open(f"{data_dir}/{sampler.__name__}_{N}_{prd}_{random_extension}.dat", "w")

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
        prd_iterations=prd,
        prd_gamma=1e-10,
    )

    exploit = get_exploitability(env, n_players, PSRO)

    info = {
        "type": "init",
        "meta_strategy": "prd",
        "meta_game": pkl.dumps(PSRO.get_meta_game()),
        "meta_strategies": pkl.dumps(PSRO.get_meta_strategies()),
        "policies": pkl.dumps(PSRO.get_policies()),
        "starting_exploit": exploit,
    }
    log_file.write(str(info) + "\n")
    history = initialize_history(PSRO, N)

    for it in tqdm.tqdm(range(steps)):
        info = debug_iteration(PSRO, sampler, N, history)
        info["it"] = it

        exploit = get_exploitability(env, n_players, PSRO)
        info["exploit"] = exploit

        log_file.write(str(info) + "\n")
        log_file.flush()

    log_file.close()


def main(prd, N, data_dir, sampler):

    while True:
        generate_data(prd, N, data_dir, sampler)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prd", default=10000, type=int, help="Number of PRD steps")
    parser.add_argument("--N", default=10, type=int, help="Samples Per Interaction")
    parser.add_argument("--sampler", default="baseline_uniform", help="Name of sampler")
    parser.add_argument("--dir", default="data", help="output_directory")
    args = parser.parse_args()

    sampler = None
    if args.sampler == "baseline_uniform":
        sampler = baseline_uniform
    elif args.sampler == "ucb":
        sampler = ucb
    elif args.sampler == "simple_ucb":
        sampler = simple_ucb

    main(args.prd, args.N, args.data, sampler)
