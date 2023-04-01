import numpy as np
import matplotlib.pyplot as plt

paths = ["rew_01_092556.npy", "rew_01_094010.npy","rew_01_095247.npy","rew_01_100904.npy","rew_01_102300.npy"]

all_rewards = np.array([np.load(f"runs/book/{path}") for path in paths])
all_rewards = np.load(f"runs/book/experiment_01_103843.npy")
plt.plot(np.mean(all_rewards,axis=0))
new_rewards = []
for old in all_rewards:
    new_rewards.append(old)
    old = old.tolist()
    mixed = old[25::-1]
    mixed += old[:25:-1]
    new_rewards.append(mixed)

new_rewards = np.array(new_rewards)
plt.plot(np.mean(new_rewards,axis=0))
plt.show()
# maybe too symmetric?
