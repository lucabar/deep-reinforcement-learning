import numpy as np
import matplotlib.pyplot as plt
run = 1

rewards = np.load(f'runs/tmp_curve.npy')

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
rewards = np.mean(rewards.reshape(-1, 10), axis=1)
simple_plot(rewards, file='Reward')
