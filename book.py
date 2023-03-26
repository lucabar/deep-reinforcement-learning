import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque # for experience replay

### ideas
## hyperparameters
replay_buffer = deque(maxlen=4000) # default:2000 is the maximum number of transitions we want to store 

## architecture 
# try only one layer

# constants / initializations
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error
total_rewards = []
target_update_freq = 10

env = gym.make("CartPole-v1")

input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

def build():
    #1 decrease replay
    replay_buffer = deque(maxlen=5000)
    #2 change learning rate 0.05
    optimizer = keras.optimizers.Adam(learning_rate=0.05)
    loss_fn = keras.losses.mean_squared_error
    model = keras.models.Sequential([
        # use 64 neurons in only one layer
        # use 512, 256, 64
    keras.layers.Dense(128, activation="elu", input_shape=input_shape),
    #keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
    # tune (output) activation relu or tanh maybe, output linear
    ])
    model.compile(optimizer, loss_fn)
    target = keras.models.clone_model(model)
    target.set_weights(model.get_weights())
    return model, optimizer, replay_buffer, target

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
            Q_values = model.predict(state[np.newaxis],verbose=0) # outputs two Q-values [np.newaxis, :]
            return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def training_step(batch_size, target):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target.predict(next_states, verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                        (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

eps = 800

model, optimizer, replay_buffer, target = build()
ep_rewards = []
for episode in range(eps):
    obs = env.reset()
    if episode % 20 == 0:
        print('ep: ', episode)
    R = 0

    ### one episode
    for step in range(475):
        epsilon = max(1 - episode / 500, 0.02)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        R += reward
        if done:
            break
    ###

    if episode > 50:
        training_step(batch_size, target)
    if episode % target_update_freq == 0:
        target.set_weights(model.get_weights())
    ep_rewards += [R]
    if episode % 100 == 0:
        np.save(f"runs/book/tmp_rew.npy",np.array(ep_rewards))

stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))
np.save(f"runs/book/rewards_{stamp}.npy",ep_rewards)
