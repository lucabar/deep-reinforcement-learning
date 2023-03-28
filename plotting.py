#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

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

args = sys.argv[1:]

run = 1
path = args[0]
count = 25

while True:
    count += 1
    try:
        rewards = np.load(f'{str(path)}{count}.npy')
    except:
        break

    rewards_conv = np.mean(rewards.reshape(-1, 10), axis=1)

    plt.title(f"Experiment No.{count}")
    plt.plot(rewards)
    plt.savefig(f"runs/book/plots/exp1_count{count}.pdf")
    # plt.show()
    plt.title(f"Experiment No.{count} (convoluted)")
    plt.plot(rewards_conv)
    # plt.show()
