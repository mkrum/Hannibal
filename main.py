
import random
from collections import deque
from abc import ABC
from dataclasses import dataclass

import pyspiel
import torch
import numpy as np

from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def sample_chance_outcomes(outcomes):
    actions, probs = zip(*outcomes)
    action = np.random.choice(actions, p=probs)
    return action


def sample_action(prob_dict):
    probs = list(prob_dict.values())
    actions = list(prob_dict.keys())
    action_id = int(np.random.choice(len(probs), p=probs))
    return actions[action_id]


def masked_row_select(tensor, mask):
    return tensor[mask.bool().unsqueeze(-1)]


@dataclass(frozen=True)
class RolloutTensor:

    state: torch.FloatTensor
    action: torch.IntTensor
    action_mask: torch.BoolTensor
    reward: torch.FloatTensor
    done: torch.BoolTensor

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None)

    def __len__(self):
        return self.state.shape[0]

    def is_empty(self):
        return self.state == None

    def add(self, state, action, action_mask, reward, done) -> "RolloutTensor":
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)
        reward = reward.unsqueeze(0)
        done = done.unsqueeze(0)

        if self.is_empty():
            return RolloutTensor(state, action, action_mask, reward, done)

        new_state = torch.cat((self.state, state), 0)
        new_action = torch.cat((self.action, action), 0)
        new_action_mask = torch.cat((self.action_mask, action_mask), 0)
        new_reward = torch.cat((self.reward, reward), 0)
        new_done = torch.cat((self.done, done), 0)
        return RolloutTensor(new_state, new_action, new_action_mask, new_reward, new_done)

    def stack(self, other) -> "RolloutTensor":

        if self.is_empty():
            return other
        
        new_state = torch.cat((self.state, other.state), 0)
        new_action = torch.cat((self.action, other.action), 0)
        new_action_mask = torch.cat((self.action_mask, other.action_mask), 0)
        new_reward = torch.cat((self.reward, other.reward), 0)
        new_done = torch.cat((self.done, other.done), 0)
        return RolloutTensor(new_state, new_action, new_action_mask, new_reward, new_done)

    def decay_(self, gamma) -> "RolloutTensor":
        for i in reversed(range(len(self) - 1)):
            self.reward[i] = self.reward[i] + ~self.done[i] * gamma * self.reward[i + 1]

    def raw_rewards(self):
        return self.reward[self.done]

    def to(self, device):
        new_state = self.state.to(device)
        new_action = self.action.to(device)
        new_action_mask = self.action_mask.to(device)
        new_reward = self.reward.to(device)
        new_done = self.done.to(device)
        return RolloutTensor(new_state, new_action, new_action_mask, new_reward, new_done)


def tensor_rollout(game, *players):
    assert game.num_players() == len(players)

    state = game.new_initial_state()

    rollout_data = [RolloutTensor.empty() for _ in range(game.num_players())]

    while not state.is_terminal():

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = sample_chance_outcomes(outcomes)
        else:
            player_id = state.current_player()
            player = players[player_id]

            probs = player.action_probabilities(state)
            action = sample_action(probs)

            state_tensor = torch.FloatTensor(state.observation_tensor(player_id))
            action_tensor = torch.LongTensor([action])
            action_mask_tensor = torch.BoolTensor(state.legal_actions_mask())
            reward_tensor = torch.FloatTensor([0])
            done_tensor = torch.BoolTensor([False])

            rollout_data[player_id] = rollout_data[player_id].add(
                state_tensor, action_tensor, action_mask_tensor, reward_tensor, done_tensor
            )

        state.apply_action(action)

    for (i, r) in enumerate(state.rewards()):
        rollout_data[i].reward[-1] = r
        rollout_data[i].done[-1] = True

    return rollout_data


def rollout(game, *players):
    assert game.num_players() == len(players)

    state = game.new_initial_state()

    while not state.is_terminal():

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = sample_chance_outcomes(outcomes)
        else:
            player_id = state.current_player()
            player = players[player_id]
            probs = player.action_probabilities(state)
            action = sample_action(probs)

        state.apply_action(action)

    return state


def n_rollout(game, N, *players):
    states = []
    for _ in range(N):
        states.append(rollout(game, *players))
    return states

def n_tensor_rollout(game, N, *players):
    rollouts = [RolloutTensor.empty() for _ in range(game.num_players())]
    for _ in range(N):
        new_rollouts = tensor_rollout(game, *players)
        for (i, n) in enumerate(new_rollouts):
            rollouts[i] = rollouts[i].stack(n)

    return rollouts


def payoff(game, N, *players):
    states = n_rollout(game, N, *players)
    mean_rewards = np.mean(
        [[s.player_return(i) for s in states] for i in range(len(players))], axis=1
    )
    return mean_rewards


class PolicyModel(nn.Module, policy_lib.Policy):

    def __init__(self, game):
        super().__init__()

        input_dim = game.observation_tensor_size()

        output_actions = game.num_distinct_actions()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_actions),
        )

        self.player_ids = 0

        self._dummy_device_param = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy_device_param.device

    def forward(self, state_tensor, action_mask):

        state_tensor = state_tensor.to(self.get_device())
        action_mask = action_mask.to(self.get_device())

        out = self.model(state_tensor)
        out = out + torch.log(action_mask + 1e-45)
        return F.log_softmax(out, dim=-1)

    def loss(self, rollout:RolloutTensor, value_est=None):
        out = self.forward(rollout.state, rollout.action_mask)
        log_probs = torch.gather(out, -1, rollout.action)

        target = rollout.reward
        if value_est is not None:
            return -1 * torch.mean((target - value_est).detach() * log_probs)
        else:
            return -1 * torch.mean(target * log_probs)

    def action_probabilities(self, state, player_id=None):
        action_mask = torch.BoolTensor(state.legal_actions_mask())
        state_tensor = torch.FloatTensor(state.observation_tensor(self.player_ids))
        
        with torch.no_grad():
            probs = torch.exp(self.forward(state_tensor, action_mask))

        actions = state.legal_actions()
        prob_dict = {a: float(probs[a].item()) for a in actions}

        # this is dangerous
        total = sum(list(prob_dict.values()))
        prob_dict = {a: float(probs[a].item()) / total for a in actions}
        return prob_dict

class ValueModel(nn.Module):

    def __init__(self, game):
        super().__init__()

        input_dim = game.observation_tensor_size()

        output_actions = game.num_distinct_actions()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self._dummy_device_param = nn.Parameter(torch.empty(0))
        self.loss_fn = nn.MSELoss()

    def get_device(self):
        return self._dummy_device_param.device

    def forward(self, state_tensor):
        state_tensor = state_tensor.to(self.get_device())
        return self.model(state_tensor)

    def loss_and_est(self, rollout):
        values = self.model(rollout.state)
        value_loss = self.loss_fn(values, rollout.reward)
        return value_loss, values.detach()



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

history = deque(maxlen=1000)
game = pyspiel.load_game("leduc_poker")

value_model = ValueModel(game)
model = PolicyModel(game)
model = model.to(device)
value_model = value_model.to(device)

opt = optim.Adam(list(model.parameters()) + list(value_model.parameters()), lr=1e-3)


def self_play(n_games, game, model):
    players = [model, model]

    rollouts = n_tensor_rollout(game, n_games, *players)

    rollout = rollouts[0].stack(rollouts[1])
    rollout.decay_(0.99)
    rollout = rollout.to(device)
    return rollouts

def vs_random(n_games, game, model):
    players = [model, policy_lib.UniformRandomPolicy(game)]

    rollouts = n_tensor_rollout(game, n_games, *players)

    rollout = rollouts[0]
    rollout.decay_(0.99)
    rollout = rollout.to(device)
    return rollouts


for i in range(10000):

    rollout = vs_random(32, game, model)

    opt.zero_grad()
    value_loss, value_est = value_model.loss_and_est(rollout)
    loss = model.loss(rollout, value_est)
    loss.backward()
    value_loss.backward()
    opt.step()

    for r in rollout.raw_rewards():
        history.append(r.cpu().item())
     
    if i % 100 == 0:
        print(np.mean(history))
        print(exploitability.nash_conv(game, model))
        print()
