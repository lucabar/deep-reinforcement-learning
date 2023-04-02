import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from plotting import convolute


def umschreib(liste: list):
    arr = np.empty([len(liste),500])
    for j in range(len(liste)):
        arr[j] = np.load(path+liste[j])
    return arr

args = sys.argv[1:]
all_data = []
path = "runs/book/"
book = {"FF":"rew_T_False_E_False_02_121425.npy","TT":"rew_T_True_E_True_02_042156.npy","FF":"rew_T_False_E_False_01_201455.npy",
        "TT":"rew_T_True_E_True_01_200921.npy","TF":"rew_T_True_E_False_01_200419.npy","FT":"rew_T_False_E_True_01_183858.npy",
        "FF":"rew_T_False_E_False_01_183333.npy","TT":"rew_T_True_E_True_01_182906.npy","TF":"rew_T_True_E_False_02_121941.npy",}
tags = ["T_True_E_True", "T_False_E_True", "T_False_E_False", "T_True_E_False"]
labels = [r"$+TN +ER$", r"$-TN -ER$", r"$+TN -ER$", r"$-TN +ER$"]

TT = umschreib(["rew_T_True_E_True_01_200921.npy","rew_T_True_E_True_02_042156.npy","rew_T_True_E_True_01_182906.npy"])
FF = umschreib(["rew_T_False_E_False_02_121425.npy","rew_T_False_E_False_01_201455.npy","rew_T_False_E_False_01_183333.npy"])
FT = umschreib(["rew_T_False_E_True_01_183858.npy"])
TF = umschreib(["rew_T_True_E_False_01_200419.npy","rew_T_True_E_False_02_121941.npy"])
plt.title("Ablation study with and without TN and ER")
rewards = [np.mean(combo, axis=0) for combo in [TT,FF,FT,TF]]
colors = ['b','orange','g','r']
for j, plot in enumerate(rewards):
    plt.plot(plot[:-10],alpha=0.2,color=colors[j])
    plt.plot(convolute(plot)[:-5],label=labels[j])
plt.xlabel(r"Episodes")
plt.ylabel(r"Reward")
plt.legend()
plt.savefig("ablation.png", dpi=400)
plt.show()

"""
for j, plot in enumerate(rewards):
    plt.plot(convolute(plot),label=labels[j])
plt.legend()
plt.show()
"""
