#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

args = sys.argv[1:]

run = 1
path = args[0]
rewards = np.load(f'{str(path)}')

def simple_plot(y_vals, x_vals = None, file = None):
    if x_vals:
        plt.plot(x_vals, y_vals)
    else:
        plt.plot(y_vals)
    if type(file) == str:
        plt.title(file)
        plt.savefig(f'runs/{file}.pdf')
    plt.show()
    return

print(len(rewards))
rewards_conv = np.mean(rewards.reshape(-1, 10), axis=1)
simple_plot(rewards)
simple_plot(rewards_conv)
