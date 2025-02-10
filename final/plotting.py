#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys


def convolute(array: np.array, dim: int = 10):
    kernel_size = dim
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(array, kernel, mode='same')

args = sys.argv[1:]

run = 1
count = 0


if __name__ == "__main__":
## file to plot 
    rewards = []
    titles = []
    for arg in args:
        try:
            rewards.append(np.load(f'{arg}'))
            titles.append(arg)
        except:
            continue

    
    colors = ['b','orange','g','r','k','magenta','cyan']
    for j, reward in enumerate(rewards):
        avg = round(np.mean(reward),3)
        median = np.median(reward)
    
        print(f"Average performance: {avg}")
        # if "mean" in args:
        #     reward = np.mean(reward,axis=0)
        # plt.title(titles[j])
        plt.plot(reward,alpha=0.2,color=colors[j],label=f"Raw data {j}")
        plt.plot(convolute(reward,5),color=colors[j],label=f"Convolution {j}")  # end always goes down due to conv
        plt.axhline(avg, color=colors[j],linewidth=0.5,label= "Average")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    #plt.savefig(f"runs/book/plots/final_performance.png",dpi=400)
    plt.show()
