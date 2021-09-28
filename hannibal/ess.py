"""
Implementation of Erdos-Selfridge-Spencer Games, see:
https://arxiv.org/pdf/1711.02301.pdf

Specifically, the Attacker-Defender Game. The api is heavily inspired by the
OpenSpiel API: https://github.com/deepmind/open_spiel
"""
from dataclasses import dataclass
from typing import Any, List

import torch
import numpy as np


def phi(state):
    K = state.tensor.shape[-1]
    w = torch.tensor([2 ** (-1 * (K - l)) for l in range(K)])
    w = w.view(1, -1)
    return torch.matmul(state.tensor, torch.transpose(w, 0, 1))


def destory_mask(state, ls, actions):
    mask = torch.zeros_like(state)

    for (i, l) in enumerate(ls):
        if actions[i] == 1:
            mask[i, l:] = 1
        else:
            mask[i, :l] = 1

        if state[i, l] != 0:
            new_split_size = torch.floor(state[i, l] / 2) + 1
            mask[i, l] = new_split_size / state[i, l]

    return mask


def n_pieces(state_tensor: torch.Tensor) -> int:
    return torch.sum(state_tensor, dim=-1)


def build_state(K, N):
    state = torch.zeros((1, K))
    idxes = torch.randint(low=0, high=K - 1, size=(N,))

    for idx in idxes:
        state[0, idx.item()] += 1.0

    return state


def is_attacker_win(state) -> bool:
    # There are pieces at the highest level
    return state.tensor[0, -1] > 0


def is_defender_win(state) -> bool:
    # There are no more pieces
    return n_pieces(state.tensor) == 0


def is_done(state):
    # Did someone win?
    return is_defender_win(state) or is_attacker_win(state)


@dataclass(frozen=True)
class AttackerState:
    tensor: torch.tensor

    def batch_size(self):
        return self.tensor.shape[0]

    def current_player(self) -> int:
        return 0

    def valid_actions(self) -> List[int]:
        return list(range(self.tensor.shape[1]))

    def random(self):
        actions = self.valid_actions()
        action = torch.tensor(
            [np.random.choice(actions) for _ in range(self.batch_size())]
        )
        return action.view(-1, 1)

    def apply_action(self, action):
        return DefenderState(self.tensor, action)


@dataclass(frozen=True)
class DefenderState:
    tensor: torch.tensor
    l: torch.tensor

    def batch_size(self):
        return self.tensor.shape[0]

    def current_player(self) -> int:
        return 1

    def valid_actions(self) -> List[int]:
        return [0, 1]

    def random(self):
        actions = self.valid_actions()
        action = torch.tensor(
            [np.random.choice(actions) for _ in range(self.batch_size())]
        )
        return action.view(-1, 1)

    def apply_action(self, action):
        K = self.tensor.shape[1]

        mask = destory_mask(self.tensor, self.l, action)

        updated_state = self.tensor * mask
        shifted_state = torch.zeros_like(updated_state)
        shifted_state[0, 1:] = updated_state[0, :-1]
        return AttackerState(shifted_state)


@dataclass(frozen=True)
class AttackerDefenderEnv:
    """
    At every timestep, the attacker proposes a partition of the current pieces
    (A and B) and then the defender selects the partition to destroy. Pieces that
    survive are then moved forward. The Attacker is trying to move a piece from
    level 0 to level N, the Defender is trying to destroy all of the pieces.
    """

    K: int  # Number of levels
    N: int  # Number of pieces
    buffer_reward: float = 0.0

    def num_players(self) -> int:
        return 2

    def reset(self) -> torch.tensor:
        state_tensor = build_state(self.K, self.N)
        return AttackerState(state_tensor)

    def step(self, state, action) -> (int, torch.Tensor, float):
        reward = {0: int(is_attacker_win(state)), 1: int(is_defender_win(state))}

        state = state.apply_action(action)

        done = is_done(state)

        new_states = build_state(self.N, self.K)
        return state, done, reward
