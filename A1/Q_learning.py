#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class QLearningAgent:

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
            prob = np.random.choice(2,size=None,p=np.array([epsilon,1-epsilon]))

            if prob == 0:
                a = np.random.randint(0,self.n_actions) # randomly chose out of 4 possible actions
            elif prob == 1:
                a = argmax(self.Q_sa[s])
            else:
                raise ValueError(f"something went wrong in choosing actions. prob={prob}")
            

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            probs = np.zeros((4))

            for (a,p) in enumerate(probs):
                probs[a] = softmax(self.Q_sa[s], temp=temp)  # create array of probability to take each action

            a = np.random.choice(4, None, p=probs)
        return a
        
    def update(self,s,a,r,s_next,done):
        G = r + self.gamma * max(self.Q_sa[s_next])
        self.Q_sa[s,a] += self.learning_rate * (G-self.Q_sa[s,a])
        return done

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    # creating instance of environment (init env - state mainly)
    env = StochasticWindyGridworld(initialize_model=False) # we have no prior access to environment
    # create instance of Agent (init Q_sa) 
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    budget = n_timesteps
    rewards = []
    playtime = []
    snap = np.arange(1,10000,200) # take snapshots of iterations
    R = 0
    s = env.start_location  # np.random.choice(env.n_states)
    start = n_timesteps

    while budget:
        '''
        if budget in snap:
            # show rendering
            print(f"@{budget}")
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=3)
        '''

        a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)  # sample action
        s_next, r, done = env.step(a)  # simulate environment (reaction/step)
        done = pi.update(s, a, r, s_next, done)  # update Q-value-table
        R += r  # append reward
        if done:
            ####  game info
            playtime.append(start-budget)
            print(f"finished after {playtime[-1]} steps.")
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=3)
            ####  game info end

            env.reset()
            s = env.start_location
            rewards += [R]
            R = 0
            start = budget-1
        else:
            s = s_next
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

        budget -=1
    return rewards, playtime


def test():
    import matplotlib.pyplot as plt
    
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards, playtime = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    game = np.arange(0,len(playtime),1)
    print(f"Obtained rewards: {rewards}.")
    plt.plot(game,playtime)
    plt.show()

if __name__ == '__main__':
    test()
