#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if (np.random.uniform(0, 1) >= epsilon):
                a = argmax(self.Q_sa[s])

            else:
                a = np.random.choice(4)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # TO DO: Add own code
            # Replace this with correct action selection
            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        return a

    def update(self, s, a, r, s_next, done):
        Gt = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s, a] = self.Q_sa[s, a] + \
            self.learning_rate * (Gt - self.Q_sa[s, a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    rewards = []

    s = env.reset()
    done = False
    for t in range(n_timesteps):
        
        # NN to provide Q values
        a = pi.select_action(s, policy, epsilon, temp)
        
        # Q[s,a1] , Q[s, a2] | exploration / exploitation
        
        # Choose action a over preffered policy
        s_next, r, done = env.step(a)
        
        # 
        pi.update(s, a, r, s_next, done)
        s = s_next

        rewards.append(r)

        if done:
            s = env.reset()

        # if plot:
        #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
        #                step_pause=0.01)

    return rewards


def test():

    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 0.1

    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate,
                         gamma, policy, epsilon, temp, plot)
    print(rewards)
    return rewards


if __name__ == '__main__':
    test()
