#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys


def simple_plot(y_vals, x_vals=None, file=None):
    if x_vals:
        plt.plot(x_vals, y_vals)
    else:
        plt.plot(y_vals)
    if type(file) == str:
        plt.title(file)
        plt.savefig(f'runs/{file}.pdf')
    plt.show()
    return

def convolute(array: np.array, dim: int = 10):
    kernel_size = dim
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(array, kernel, mode='same')

args = sys.argv[1:]

run = 1
count = 0


try:
    rewards = np.load(f'runs/book/{args[0]}')
except:
    exit()

avg = round(np.mean(rewards[-50:]),3)
# rewards_conv = np.mean(rewards.reshape(-1, 10), axis=1)
median = np.median(rewards)
plt.title(f"Experiment No.{count}")
# plt.plot(np.mean(rewards,axis=0))
plt.plot(rewards)
plt.axhline(50, color='red')
# plt.savefig(f"runs/book/plots/exp1_count{count}.pdf")
plt.show()
plt.title(f"Experiment No.{count} (convoluted)")
# plt.plot(rewards_conv)
# plt.show()
print(f'count {count}, average last 50 eps: {avg}, median: {median}')

plt.title('Convoluted')
plt.plot(convolute(rewards))
plt.show()
