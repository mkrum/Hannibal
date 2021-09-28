from hannibal.ess import *
import torch
import numpy as np

# np.random.seed(1234)
# torch.manual_seed(1234)


def test_state_tensor():

    N = 10
    K = 10
    state_tensor = build_state(10, 10)

    assert state_tensor.shape == (1, K)
    assert n_pieces(state_tensor) == N


def test_env():
    N = 10
    K = 10
    env = AttackerDefenderEnv(K, N)

    state = env.reset()
    assert state.current_player() == 0

    done = False
    while not done:
        action = state.random()
        state, done, reward = env.step(state, action)
        print(state)
        print(reward)
