#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_and_mean(files: list[str]) -> list:
    rewards = []
    for plot in files:
        rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
    return np.mean(rewards,axis=0)

def load_list(files: list[str]) -> list:
    rewards = []
    for plot in files:
        rew = np.load(f'data/rewards/r_{plot}.npy')
        rewards.append(rew)
    return np.array(rewards)

def plot_list_errors(list_strings: list[str], save_name: str = None):
    reward = load_list(list_strings)
    # Calculate the mean and standard deviation of the data
    means = np.mean(reward, axis=0)
    stds = np.std(reward, axis=0, ddof=1)
    # Calculate the standard error of the mean
    sem = stds / np.sqrt(reward.shape[1])

    # Calculate the 95% confidence interval
    conf_int = 1.96 * sem

    # Plot the means with error bars representing the confidence interval
    plt.errorbar(np.arange(reward.shape[1]), means, yerr=conf_int, fmt='o', capsize=1,color='tab:red')
    plt.plot(np.arange(reward.shape[1]),np.mean(reward,axis=0))

    plt.ylim(-10,36)
    plt.title('Default vs speed 0.5')
    plt.grid()
    plt.legend()
    if save_name:
        plt.savefig(save_name)
    plt.show()

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']


### Part 1

old_part1_full = ['27_220351','27_230853','28_002357','28_013246','28_065433']
old_part1_reinforce = ['27_182504','27_184923','27_191045','27_193144','27_195244']
old_part1_bootstrap = ['27_201503','27_205749','28_075635']
old_part1_MCbaseline = ['28_153349']

part1_full = ['04_183040','05_075841']
part1_reinforce = ['04_215854']
part1_bootstrap = ['05_034633','05_053914']
part1_MCbaseline = ['04_225730','05_004942']

hyperopt_big = ['21_205207','21_224331','21_205312','22_002411','22_015841','22_021350','22_042033','22_054051','22_063949']
hyperopt_big += ['22_072616','22_083833','22_101711','22_094037','22_140345','22_163358','22_180822','22_194411','22_211909']
hyperopt_big += ['22_231129','23_004433']


rewards = []
for plot in part1_reinforce:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label='REINFORCE')


rewards = []
for plot in part1_MCbaseline:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label = 'MC baseline')


rewards = []
for plot in part1_bootstrap:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label='bootstrap')


rewards = []
for plot in part1_full:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy')[:250])
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label = 'full')

plt.title('End of Part 1')
plt.grid()
plt.legend()
plt.savefig('plots/Part_1.pdf')
plt.show()

learning_plots = ['23_131422','23_151619','23_203642','23_225132']  # with 0.01 eta
eta_plots = ['26_171012','26_151751','26_175224']  # with 0.001 learning '25_181333','25_201320','25_221342' more runs but bad
# 0.5, 0.1, 0.01, 0.001, 0.0005, 0.0001
tuning = eta_plots + learning_plots

labels = [r'$\eta=0.001$',r'$\eta=0.0005$',r'$\eta=0.0001$',
          r'$\alpha=0.1$',r'$\alpha=0.01$',r'$\alpha=0.001$',r'$\alpha=0.0001$']
for j, plots in enumerate(tuning):
    linestyle = "-"
    if j > 2:
        linestyle = "--"
    reward = np.load(f'data/rewards/r_{plots}.npy')
    reward = savgol_filter(reward,10,1)
    plt.plot(reward, label=labels[j], linestyle=linestyle)
plt.title('Hyperparameter tuning')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='right')  # fontsize='8'
plt.savefig('plots/learning-eta_tuning.pdf')
plt.show()

# for j, plots in enumerate(eta_plots):
#     linestyle = "-"
#     labl = f'{j}:' + plots
#     reward = np.load(f'data/rewards/r_{plots}.npy')
#     reward = savgol_filter(reward,10,1)
#     plt.plot(reward, label=labl, linestyle=linestyle)
# plt.title('Eta hyperopt')
# plt.grid()
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend()
# plt.savefig('plots/eta_hyperopt.pdf')
# plt.show()


### PART 2
default = ['30_202045','30_204133','30_210223','30_212339','01_065527']
speed_05 = ['30_113517','30_115449','30_121431','30_123431','30_140008']
speed_15 = ['30_151006','30_151743','30_152617','30_153421','30_142345']
speed_20 = ['30_154243','30_154719','30_155202','30_155709']
