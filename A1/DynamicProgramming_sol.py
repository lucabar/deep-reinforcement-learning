#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.threshold = threshold
        self.name = 'loc1'
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s,a] = p_sas[s,a]@(r_sas[s,a]+self.gamma*self.Q_sa.max(axis=1))
        return

    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
    max_error = 1
    count = 1
    while max_error > threshold:
        max_error = 0
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                Q = QIagent.Q_sa[s,a]
                QIagent.update(s,a,env.p_sas,env.r_sas)
                max_error = max(abs(QIagent.Q_sa[s,a]-Q),max_error)
        #print(QIagent.Q_sa[52])
        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        if count == 1:
            env.save(f'start-pic-{QIagent.name}')
        elif count == 9:
            env.save(f'mid-pic-{QIagent.name}')
        #print("Q-value iteration, iteration {}, max error {}".format(count,max_error))
        count +=1
    #print(count)
    env.save(f'end-pic-{QIagent.name}')

    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    # View optimal policy
    QIagent = Q_value_iteration(env,gamma,threshold)  # let agent figure out best policy a priori (get best Q_sa)

    # pre-game policy deciding is finished. Now to the experiment(s)    

    V_3 = max(QIagent.Q_sa[3])  # optimal value at start s=3 loc=(0,3)
    print(f"Converged optimal value at start (0,3): V*(s=3)= {round(V_3,1)}.")

    
    exp_step = np.ceil(env.goal_rewards[0] - V_3 + 1)
    print(f"expected average # of steps: {exp_step}")
    mean_reward_per_timestep = (V_3)/exp_step
    # of course there exist no half steps (or .2), so the outcome of this will have to be
    # rounded up for the game to finish. the rounded down value can not possibly be enough

    Ret = 0
    reps = 5  # amnt of repitions of experiment
    count = np.zeros((reps))  # counter of how long rep went

    for k in range(reps):
        done = False
        print('Reset!')
        s = env.reset()
        while not done:
            count[k] += 1
            a = QIagent.select_action(s)
            s_next, r, done = env.step(a)
            #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            s = s_next
            Ret += r  # this includes final reward of +40
        print(f"actual counts: {count}")
        print(f"actual avg count: {np.sum(count)/(k+1)}")

    print(f"Mean reward per timestep under optimal policy: {round(mean_reward_per_timestep,2)}")
    return mean_reward_per_timestep

if __name__ == '__main__':
    experiment()
