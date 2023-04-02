from dqn import learning
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


'''
expected comand line arguments:
--target_active
--experience_replay
'''

args = sys.argv[1:]
target_active, replay_active = False, False
np.random.seed(42)


eps = 50
n_runs = 10

all_rewards = np.empty([n_runs,eps])
training = False

policy = "greedy"
temps = [0.5]
path_to_weights = None


# w/out training
if not training:
    path_to_weights = "insane_start_w_01_112419.h5"
    # path_to_weights = "w_01_143105.h5"
    policy = "greedy"


## hyperparameters
learning_rate, batch_size, arch, target_update_freq, replay_buffer_size = (0.0001, 32, 1, 10, 5000)
freqs = [10]
buffer_sizes = [5000]
temps = [1,5,20]
epsilon = 0.8
arch = 1
temp = 1

try:
    for arg in args:
        if arg == "--target_active":
            target_active = True
        elif arg == "--experience_replay":
            replay_active = True
except:
    pass

combs = [(True,True)]

for run in range(n_runs):
    for comb in combs:
        for temp in temps:
            for replay_buffer_size in buffer_sizes:
                rewards = learning(eps, learning_rate, batch_size, architecture=arch, target_update_freq=target_update_freq, 
                                replay_buffer_size=replay_buffer_size, policy=policy, epsilon=epsilon, 
                                path_to_weights=path_to_weights, temp=temp,replay_active=comb[0],target_active=comb[1], double_dqn=False)
                stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))
                np.save(f'runs/book/rew_final_play_{run}',rewards)

# np.save(f"runs/book/experiment_{stamp}", all_rewards)
