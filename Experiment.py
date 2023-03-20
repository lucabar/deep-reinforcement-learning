#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from assignment_2 import q_learning
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy',
                             epsilon=None, temp=None, smoothing_window=51, plot=False, n=5):

    reward_results = np.empty([n_repetitions, n_timesteps])  # Result array
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate,
                                 gamma, policy, epsilon, temp, plot)
        reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    # average over repetitions
    learning_curve = np.mean(reward_results, axis=0)
    # additional smoothing
    learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve


def experiment():
    # Settings
    # Experiment
    n_repetitions = 50
    smoothing_window = 1001
    plot = False  # Plotting is very slow, switch it off when we run repetitions

    # MDP
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
    # Parameters we will vary in the experiments, set them to some initial values:
    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q'  # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25

    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning'}

    # Assignment 2: Effect of exploration
    policy = 'egreedy'
    # epsilons = [0.02, 0.1, 0.3]
    learning_rate = 0.25
    backup = 'q'
    Plot = LearningCurvePlot(
        title='Exploration: $\epsilon$-greedy versus softmax exploration')

    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot)
    Plot.add_curve(
        learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))

    Plot.save('q-learning.png')


if __name__ == '__main__':
    experiment()
