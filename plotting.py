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
count = 0

while count < 1:
    try:
        count += 1
        rewards = np.load(f'runs/book/{args[0]}')
    except:
        break
    avg = round(np.mean(rewards[-100:]),3)
    # rewards_conv = np.mean(rewards.reshape(-1, 10), axis=1)
    median = np.median(rewards)
    plt.title(f"Experiment No.{count}")
    plt.scatter(np.arange(1,len(rewards)+1,1),rewards)
    plt.axhline(50)
    # plt.savefig(f"runs/book/plots/exp1_count{count}.pdf")
    plt.show()
    plt.title(f"Experiment No.{count} (convoluted)")
    # plt.plot(rewards_conv)
    plt.show()
    print(f'count {count}, average last 100 eps: {avg}, median: {median}')
