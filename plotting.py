#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from Helper import convolute

args = sys.argv[1:]

run = 1
count = 0

if __name__ == "__main__":
    try:
        rewards = np.load(f'runs/book/{args[0]}')
    except:
        exit()

    avg = round(np.mean(rewards),3)
    median = np.median(rewards)
    print(f"avg: {avg}")
    #plt.title(f"Final Cart Pole performance")
    if "mean" in args:
        rewards = np.mean(rewards,axis=0)
    plt.plot(rewards,alpha=0.2,color='b',label="Raw data")
    plt.plot(np.arange(3,len(rewards)-3,1),convolute(rewards)[3:-3],color='b',label="Convolution of average (width 10)")  # end always goes down due to conv
    plt.axhline(avg, color='red',linewidth=0.5,label= "Average")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    #plt.savefig(f"runs/book/plots/final_performance.png",dpi=400)
    plt.show()
