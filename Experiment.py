#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from poledance import q_learning
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(eps, n_repetitions, learning_rate,
                             epsilon=None, temp=None, smoothing_window=5):

    reward_results = np.empty([n_repetitions, eps])  # Result array
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        print(f'Rep. number {rep+1}')
        rewards = q_learning(eps, learning_rate, epsilon=epsilon, temp=temp)
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
    n_repetitions = 5

    # Parameters we will vary in the experiments, set them to some initial values:
    # Exploration
    epsilon = 0.01
    temp = 1.0
    # Back-up & update
    learning_rate = 0.001
    # hyperparameters
    # learning_rate = [0.0001, 0.001, 0.01]
    # batch_size =  [32, 64, 128]
    # update_target_freq = [100, 500, 1000]

    Plot = LearningCurvePlot(
        title='First experiments; testing...')

    learning_curve = average_over_repetitions(100, n_repetitions, learning_rate, epsilon=epsilon)
    stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))

    np.save(f'runs/tmp_curve', learning_curve)
    Plot.add_curve(
        learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    Plot.save(f'q-learning{stamp}.png')


if __name__ == '__main__':
    experiment()
