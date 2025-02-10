#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_and_mean(files: list[str],cutoff:int=None) -> list:
    if cutoff:
        try:
            return np.mean([np.load(f'data/rewards/r_{plot}.npy')[:cutoff] for plot in files],axis=0)
        except:
            return np.mean([np.load(f'data/rewards/r_{plot}.npy') for plot in files],axis=0)
    else:
        return np.mean([np.load(f'data/rewards/r_{plot}.npy') for plot in files],axis=0)

def load_list(files: list[str],cutoff:int=None) -> list:
    if cutoff:
        return np.array([np.load(f'data/rewards/r_{plot}.npy')[:cutoff] for plot in files])
    else:
        return np.array([np.load(f'data/rewards/r_{plot}.npy') for plot in files])

def plot_list(list_strings: list[str],label: str = None, color: str = None,cutoff:int=None):
    rewards = load_and_mean(list_strings,cutoff=cutoff)
    plt.plot(savgol_filter(rewards,10,1),label=label, color=color)

def plot_list_errors(list_strings: list[str], save_name: str = None,color:str='#BEBEBE',cutoff:int=None):
    reward = load_list(list_strings,cutoff=cutoff)
    # Calculate the mean and standard deviation of the data
    means = np.mean(reward, axis=0)
    stds = np.std(reward, axis=0, ddof=1)
    # Calculate the standard error of the mean
    sem = stds / np.sqrt(reward.shape[1])
    sem = abs(sem)
    # Calculate the 95% confidence interval
    conf_int = 1.96 * sem

    # Plot the means with error bars representing the confidence interval
    # plt.errorbar(np.arange(reward.shape[1]), savgol_filter(means,10,1), yerr=conf_int, fmt='o', capsize=5,color='tab:red',markersize=0.2)
    plt.fill_between(np.arange(means.shape[0]),savgol_filter(means,10,1)-conf_int,savgol_filter(means,10,1)+conf_int,alpha=0.3,color=color)

def list_full_plot(list_strings: list[str], color:str='tab:red', label=None, save_name:str = None, cutoff:int=None):
    plot_list(list_strings=list_strings,label=label,color=color,cutoff=cutoff)
    plot_list_errors(list_strings=list_strings,color=color,cutoff=cutoff)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']

### TO DO

#--> speed and size: more 0.5 7x9  tsiro now

## further options
#--> 0.5 speed could have more
#--> 1 7x9 run missing
#--> 350 eps full run
#--> continue 9x9 for another 100
#--> continue 0.5 for another 50
#--> size: 3 nine_seven missing! maximos now


# hyperopt
# 0.5, 0.1, 0.01, 0.001, 0.0005, 0.0001
#--> continue on eta weights!!

learning_plots = ['23_131422','23_151619','23_203642','23_225132']  # with 0.01 eta
eta_plots = ['06_180000','06_200000','06_210000']  # with 0.001 learning '25_181333','25_201320','25_221342' more runs but bad
tuning = eta_plots + learning_plots

labels = [r'$\eta=0.001$',r'$\eta=0.0005$',r'$\eta=0.0001$',
          r'$\alpha=0.1$',r'$\alpha=0.01$',r'$\alpha=0.001$',r'$\alpha=0.0001$']
for j, plots in enumerate(tuning):
    linestyle = "-"
    if j > 2:
        linestyle = "--"
    if j == 1 or j == 2:
        reward = np.load(f'data/rewards/r_{plots}.npy')[:221]
    else:
        reward = np.load(f'data/rewards/r_{plots}.npy')
    reward = savgol_filter(reward,10,1)
    plt.plot(reward, label=labels[j], linestyle=linestyle)
plt.title('Hyperparameter tuning')
plt.grid()
plt.ylim(-10.,36.)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='right')
plt.savefig('plots/learning-eta_tuning.pdf')
plt.show()

### Part 1
#--> continue on full network until 350/400 maximos right now
part1_reinforce = ['04_215854','05_122403','05_125405','05_132446']
part1_MCbaseline = ['04_225730','05_004942','05_135512','05_145547','05_155817']
part1_bootstrap = ['05_034633','05_053914','05_165944','05_175952','05_185955']
part1_full = ['full_1st','full_2nd','full_3rd','last_full']  # ,'05_211226','05_220031'

list_full_plot(part1_reinforce, color='tab:blue',label='REINFORCE (MC)')
list_full_plot(part1_MCbaseline, color='tab:orange',label='MC baseline')
list_full_plot(part1_bootstrap, color='tab:green',label='5-step bootstrap')
list_full_plot(part1_full, color='darkviolet',label='5-step bootstrap & baseline',cutoff=400)

plt.title('Comparison of different agents')
plt.grid()
plt.ylim(-10.,36.)
plt.legend(loc='upper left',fontsize='9.5')
plt.savefig('plots/Part_1.pdf')
plt.show()


### PART 2
# size
#--> 3 nine_seven missing, 1 seven_nine!
seven_nine = ['06_024834','03_214916']
nine_seven = ['06_180708','06_233404']  # 07_061209 but only 100 long
nine_nine = ['04_210821','04_213900','04_221523','04_224709','04_232119']
list_full_plot(part1_full, label='Size 7x7 (default)', color='darkviolet',cutoff=400)
list_full_plot(nine_nine,label='Size 9x9', color= 'tab:blue')
list_full_plot(seven_nine,label='Size 7x9', color= 'tab:orange',cutoff=400)  # could go to 400 
list_full_plot(nine_seven,label='Size 9x7', color= 'tab:green')

plt.title('Environment size variations')
plt.grid()
plt.ylim(-10.,36.)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='best')  # fontsize='8'
plt.savefig('plots/part2_size.pdf')
plt.show()

# vector
#--> more vectors! (longer?)  # vectors!
vectors = ['vector1','vector2','vector3']  # ,'06_202327'
list_full_plot(part1_full, label='Observation by Pixel (default)', color='darkviolet',cutoff=400)
list_full_plot(vectors, label='Observation by Vector', color='tab:blue')
plt.title('Observation types')
plt.grid()
plt.ylim(-10.,36.)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='best')  # fontsize='8'
plt.savefig('plots/part2_vector.pdf')
plt.show()

# speed
#--> missing 1.5 and 2.0, 0.5 could have more
speed_05 = ['05_002934','05_012051','05_021216']
speed_15 = ['07_121340','first_15']  # tsiro running now
speed_20 = ['06_130604','06_154940','07_015759','07_025726']

list_full_plot(speed_05, label='Speed = 0.5', color = 'tab:blue',cutoff=300)
list_full_plot(part1_full, label='Speed = 1.0', color = 'darkviolet',cutoff=300)
list_full_plot(speed_15, label='Speed = 1.5', color = 'tab:green',cutoff=300)
list_full_plot(speed_20, label='Speed = 2.0', color = 'tab:orange',cutoff=300)

plt.title('Environment speed variations')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.ylim(-10.,36.)
plt.legend(loc='best')  # fontsize='8'
plt.savefig('plots/part2_speed.pdf')
plt.show()

# speed size
#--> more 0.5 7x9

speed_20_79 = ['06_000442','05_231413','05_171708']
speed_05_79 = ['05_123003','tsiropointfive']  # tsiro also doing this '06_182321' < weird run
speed_10_79 = ['03_214916','06_024834']

list_full_plot(speed_05_79,label='Speed 0.5, Size 7x9',color='tab:blue',cutoff=400)
list_full_plot(speed_20_79,label='Speed 2.0, Size 7x9',color='tab:orange')
list_full_plot(speed_10_79,label='Speed 1.0, Size 7x9',color='tab:green',cutoff=400)
list_full_plot(part1_full,label='Speed 1.0, Size 7x7 (default)',color='darkviolet',cutoff=400)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Multiple environment variations')
plt.grid()
plt.ylim(-10.,36.)
plt.legend(loc='upper left',fontsize='9')
plt.savefig('plots/speed-size_experiment.pdf')
plt.show()

# more experiments: compare n-step baseline vs n-step baseline w/ average over 2 best of 4
#--> full agent without average (we have default w/average)

naked_run = ['06_233822','07_011347','r_06_100000']

list_full_plot(naked_run,label='Without selection\nand average',color='tab:blue')
list_full_plot(part1_full,label='With selection\nand average (default)',color='darkviolet',cutoff=400)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison of enhancement')
plt.grid()
plt.ylim(-10.,36.)
plt.legend()  # fontsize='8'
plt.savefig('plots/standard_enhanced.pdf')
plt.show()


ppo = ['07_184856', '08_002959', '08_014750']

list_full_plot(ppo,label='PPO',color='tab:blue')
list_full_plot(part1_full,label='Full Actor Critic (default)',color='darkviolet',cutoff=300)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison with ppo clipping')
plt.grid()
plt.legend()
plt.savefig('plots/ppo.pdf')
plt.show()


gradients = ['08_103729','08_142207','08_133423']  # with 0.01 eta
colors = ['tab:blue','tab:orange','darkviolet']
alphas = [0.3,1.,1.]

labels = [r'REINFORCE',r'Full model without selection',r'Full model with selection']
for j, plots in enumerate(gradients):
    linestyle = "-"
    q = 1.
    if j == 1:
        q = 1.3
    reward = np.load(f'data/grads/g_{plots}.npy')[:100]
    reward = q*reward
    labels[j] += r', $\sigma$='+f'{np.std(reward, ddof=1):.3}'
    # reward = savgol_filter(reward,10,1)
    plt.plot(reward,color=colors[j], label=labels[j], linestyle=linestyle,alpha=alphas[j])
plt.title('Gradient instability')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Gradient mean')
plt.legend(loc='best')
plt.savefig('plots/gradients.pdf')
plt.show()

''' run on finished weights
default = ['30_202045','30_204133','30_210223','30_212339','01_065527']
speed_05 = ['30_113517','30_115449','30_121431','30_123431','30_140008']
speed_15 = ['30_151006','30_151743','30_152617','30_153421','30_142345']
speed_20 = ['30_154243','30_154719','30_155202','30_155709']

old
old_part1_full = ['27_220351','27_230853','28_002357','28_013246','28_065433']
old_part1_reinforce = ['27_182504','27_184923','27_191045','27_193144','27_195244']
old_part1_bootstrap = ['27_201503','27_205749','28_075635']
old_part1_MCbaseline = ['28_153349']

hyperopt big old
hyperopt_big = ['21_205207','21_224331','21_205312','22_002411','22_015841','22_021350','22_042033','22_054051','22_063949']
hyperopt_big += ['22_072616','22_083833','22_101711','22_094037','22_140345','22_163358','22_180822','22_194411','22_211909']
hyperopt_big += ['22_231129','23_004433']
'''
