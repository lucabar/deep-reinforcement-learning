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
from Helper import softmax, argmax
import matplotlib.pyplot as plt

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.uniform(0,1) <= epsilon:
                a = np.random.randint(0,self.n_actions) # randomly chose out of 4 possible actions
            else:
                a = argmax(self.Q_sa[s])

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            probs = softmax(self.Q_sa[s,np.arange(4)], temp=temp)  # create array of probability to take each action
            a = np.random.choice(4, None, p=probs)
        return a
        
    def update(self,s,a,r,s_next,a_next,done):
        if done:
            pass
        _G = r + self.gamma * self.Q_sa[s_next,a_next]
        self.Q_sa[s,a] += self.learning_rate * (_G-self.Q_sa[s,a])
        return done
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    budget = n_timesteps
    rewards = []
    playtime = []
    #### take snapshots of iterations
    snap = np.arange(1,10000,200)
    ###
    s = env._location_to_state(env.start_location)  # sample state as integer, not loc!
    a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)  # sample action

    start = n_timesteps

    while budget:
        '''
        if budget in snap:
            # show rendering
            print(f"@{budget}")
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=3)
        '''
        s_next, r, done = env.step(a)  # simulate environment (reaction/step)
        a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)  # sample action
        done = pi.update(s, a, r, s_next, a_next, done)  # update Q-value-table
        rewards += [r]  # append reward
        s, a = s_next, a_next

        if done:
            ####  game info
            playtime.append(start-budget)
            #print(f"finished after {playtime[-1]} steps.")
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.5)
            ####  game info end

            s = env.reset()
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)  # sample action
            start = budget-1

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

        budget -=1
    return rewards, playtime


def test():
    n_timesteps = 5000
    gamma = 1.0
    learning_rate = 0.25

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards, playtime = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print(f"Obtained rewards: {rewards}")   

    # show learning curve
    #game = np.arange(0,len(playtime),1)
    #plt.plot(game,playtime)
    #plt.show()     
    
if __name__ == '__main__':
    test()
