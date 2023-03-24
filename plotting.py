import numpy as np
import matplotlib.pyplot as plt

run = 1
path = '23_164716_l0.0025_e0.05_b32_u50_a3.npy'
rewards = np.load(f'runs/rewards1.npy')

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
