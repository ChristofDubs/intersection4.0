"""simple 2 car agent

This module contains a class that learns from interacting with a 2 car environment

author: Christof Dubs
"""
import random
import numpy as np
import context

from intersection4.environments.base.definitions import Action


class Agent:
    def __init__(self):
        self.values = {}
        self.policy = {}
        self.eps = 0.1
        self.alpha = 1
        self.gamma = 1
        self.experience = []
        self.num_action_pairs = Action.NUM_ACTIONS**2
        self.action_pairs = [None] * self.num_action_pairs
        for i in range(Action.NUM_ACTIONS):
            for j in range(Action.NUM_ACTIONS):
                action_pair = [Action(i), Action(j)]
                idx = self.convert_action_pair_to_idx(action_pair)
                self.action_pairs[idx] = action_pair

    def convert_action_pair_to_idx(self, action_pair):
        return action_pair[0] * Action.NUM_ACTIONS + action_pair[1]

    def add_values(self, key):
        assert(key[0] != -1 or key[2] != -1)
        self.values[key] = [-5000] * self.num_action_pairs

    def get_epsilon_greedy_action(self, state):
        key = tuple(state)
        if key not in self.values:
            self.add_values(key)
            # x = random.randint(0, self.num_action_pairs-1)
            # return [Action(int(x/3)),Action(x%3)]
            return [Action.ACCELERATE, Action.ACCELERATE]

        max_val = -np.inf
        best_actions = []
        for action_pair, val in enumerate(self.values[key]):
            if val > max_val:
                best_actions = [action_pair]
                max_val = val
            elif val == max_val:
                best_actions.append(action_pair)

        x = random.random()
        if len(best_actions) == self.num_action_pairs:
            return self.action_pairs[int(x * self.num_action_pairs)]

        threshold = self.eps * (self.num_action_pairs) / (self.num_action_pairs - len(best_actions))
        if x < threshold:
            return self.action_pairs[int(self.num_action_pairs * x / threshold)]

        return self.action_pairs[best_actions[int(
            len(best_actions) * (x - threshold) / (1 - threshold))]]

    # Q-learning
    def update(self, old_state, old_action, reward, new_state):
        old_state_value = self.values[tuple(old_state)][self.convert_action_pair_to_idx(old_action)]
        new_state_value = max(self.values[tuple(new_state)])
        self.values[tuple(old_state)][self.convert_action_pair_to_idx(old_action)
                                      ] += self.alpha * (reward + self.gamma * new_state_value - old_state_value)

    def save_data(self, old_state, action, reward):
        self.experience.append([old_state, action, reward])

    def learn(self):
        num_steps = len(self.experience)
        for i in range(num_steps - 1, 0, -1):
            new_state = self.experience[i][0]
            [old_state, action, reward] = self.experience[i - 1]
            self.update(old_state, action, reward, new_state)

    def on_termination(self):
        terminal_state = [-1,-1]
        self.values[tuple(terminal_state)] = [0]
        self.save_data(terminal_state, None, None)
        self.learn()
        self.experience.clear()
