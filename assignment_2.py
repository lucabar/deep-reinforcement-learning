import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    # env = StochasticWindyGridworld(initialize_model=False)
    # pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    rewards = []

    s = env.reset()
    done = False
    for t in range(n_timesteps):
        a = env.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(a)
        epsilon = epsilon * 0.95
        env.update(s, a, r, s_next, done)
        s = s_next

        rewards.append(r)

        if done:
            s = env.reset()
    return rewards