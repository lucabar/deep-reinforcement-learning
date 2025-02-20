#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tensorflow as tf


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def softmax(temp, state, network):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    s_tensor = make_tensor(state, False)
    x = network.model.predict(s_tensor, verbose=0)
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.argmax(np.exp(z/temp)/np.sum(np.exp(z/temp)))  # compute softmax


def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def convolute(array: np.array, dim: int = 10):
    kernel_size = dim
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(array, kernel, mode='same')

def make_tensor(s, list: bool):
    '''in order to be used in net.predict() method'''
    s_tensor = tf.convert_to_tensor(s)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)


def stable_loss(target, pred):  # implement own loss on stable target
    '''Squared loss'''
    squared_difference = tf.square(target - pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


def e_greedy(epsilon, state, network):
    ''' epsilon greedy policy '''
    if np.random.uniform(0., 1) > epsilon:
        s_tensor = make_tensor(state, False)
        Q_vals = network.model.predict(s_tensor, verbose=0)  # outputs two Q-values
        return np.argmax(Q_vals)
    else:
        return np.random.randint(0, 2)


def linear_anneal(t, T, start, final, percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

def find_best_reward():
    first = 0
    med_first = 0
    for count in range(285):
        try:
            rewards = np.load(f'runs/book/rew{count}.npy')
        except:
            continue
        
        avg = round(np.mean(rewards[50:]),3)
        median = np.median(rewards[50:])
        if avg > first:
            print(f"Reward: {avg} at count: {count}")
            first = avg
        if median > med_first:
            print(f"Median: {median} at count: {count}")
            med_first = median


if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4  # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y, label='method 1')
    LCTest.add_curve(smooth(y, window=35), label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')
