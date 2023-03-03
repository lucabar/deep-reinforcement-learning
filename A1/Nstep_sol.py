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

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(actions)
        for t in range(T_ep):
            m = min(self.n, T_ep-t)
            if done and states[t+m]==states[-1]:
                _G = self.gamma**np.arange(m) @ rewards[t:t+m]
            else:
                _G = self.gamma**np.arange(m)@rewards[t:t+m] + self.gamma**m * np.max(self.Q_sa[states[t+m]])
            self.Q_sa[states[t],actions[t]] += self.learning_rate * (_G-self.Q_sa[states[t],actions[t]])
        return done


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    big_R = []
    budget = n_timesteps

    while budget>=0:
        s = env.reset()
        states, actions, rewards = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=float)
        
        # collect episode
        for t in range(max_episode_length):
            a = pi.select_action(s,policy,epsilon,temp)
            s_next, r, done = env.step(a)
            states, actions, rewards = np.append(states,s), np.append(actions,a), np.append(rewards,r)
            big_R = np.append(big_R,r)
            s = s_next
            budget -=1
            if done:
                #print(f'won @ {budget, t}')
                break
        
        states = np.append(states,s)
        # update
        done = pi.update(states, actions, rewards, done)

        if plot and not np.random.randint(0,100):
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return big_R[:n_timesteps]

def test():
    import time
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
    learning_rate = 0.25
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    start = time.time()
    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print(f"Obtained rewards: {rewards}")
    print(f"it took {(time.time()-start)/60}mins.")

if __name__ == '__main__':
    test()
