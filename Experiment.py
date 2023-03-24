#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from poledance import q_learning
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(eps, n_repetitions, learning_rate,
                             epsilon=None, temp=None, smoothing_window=5, batch_size=32, update_target_freq=100):

    reward_results = np.empty([n_repetitions, eps])  # Result array
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        print(f'Rep. number {rep+1}')
        rewards = q_learning(eps, learning_rate, epsilon=epsilon, temp=temp, batch_size=batch_size, update_target_freq=update_target_freq)
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
    epsilon = 0.05
    temp = 1.0
    test_nb = 0

    # single value test
    learning_rate = 0.001
    update_target_freq = 100
    batch_size = 32

    # lists of hyperparameters
    learning_rates = [0.001, 0.0025]
    batch_sizes =  [32]
    update_target_freqs = [4]

    alternatives = [(True,True),(False,False),(True,False),(False,True)]  # tests with various target_network or experience replay

    for update_target_freq in update_target_freqs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                test_nb += 1
                print(f"-----Testing no.{test_nb}, l{learning_rate}, b{batch_size}, u{update_target_freq}-----")

                Plot = LearningCurvePlot(
                    title=f'Learning rate: {learning_rate}, batch size: {batch_size}, target update: {update_target_freq}')

                learning_curve = average_over_repetitions(200, n_repetitions, learning_rate, epsilon=epsilon, batch_size=batch_size, update_target_freq=update_target_freq)
                stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))

                np.save(f'runs/data/{stamp}_l{learning_rate}_e{epsilon}_b{batch_size}_u{update_target_freq}_a3', learning_curve)
                Plot.add_curve(
                    learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
                Plot.save(f'runs/{stamp}_l{learning_rate}_e{epsilon}_b{batch_size}_u{update_target_freq}_a3.pdf')
                print(f"-----Test {test_nb}, l{learning_rate}, b{batch_size}, u{update_target_freq} done!-----")


if __name__ == '__main__':
    experiment()
