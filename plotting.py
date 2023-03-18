import numpy as np
import matplotlib.pyplot as plt

losses = np.load('runs/all_losses.npy')
rewards = np.load('runs/all_ep_rewards.npy')

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

simple_plot(losses, file='Loss')
simple_plot(rewards, file='Reward')
