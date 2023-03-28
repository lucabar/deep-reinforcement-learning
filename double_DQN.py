import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque # for experience replay

### ideas
## hyperparameters
replay_buffer = deque(maxlen=200) # default:2000 is the maximum number of transitions we want to store 

## architecture 
# try only one layer

# constants / initializations
batch_size = 32
discount_factor = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.mean_squared_error
total_rewards = []
target_update_freq = 10

env = gym.make("CartPole-v1")


input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

def build_model(j: int = 1, activ: str = "elu"):
    if j == 1:
        model = keras.models.Sequential([
        # use 64 neurons in only one layer
        # use 512, 256, 64
        keras.layers.Dense(64, activation=activ, input_shape=input_shape),
        #keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(n_outputs)
        # tune (output) activation relu or tanh maybe, output linear
        ])
    elif j == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=activ, input_shape=(4,)),
            tf.keras.layers.Dense(2)
        ])
    elif j == 3:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=activ, input_shape=(4,)),
            tf.keras.layers.Dense(32, activation=activ),
            tf.keras.layers.Dense(2)
        ])
    elif j == 4:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation=activ, input_shape=(4,)),
            tf.keras.layers.Dense(16, activation=activ),
            tf.keras.layers.Dense(16, activation=activ),
            tf.keras.layers.Dense(2)
        ])
    return model


def build(j: int = 1, lr: float = 0.01):
    #1 decrease replay
    #2 change learning rate 0.05
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.mean_squared_error
    model = build_model(j)
    model.compile(optimizer, loss_fn)
    target = keras.models.clone_model(model)
    target.set_weights(model.get_weights())
    return model, optimizer, target

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

    # --------------------- Double DQN --------------------- #
    # Now we use the online model to predict the Q-values and not the target model
    # This is due to the fact that the target network overestimates the Q-values
    next_Q_values = model.predict(next_states, verbose=0) 

    best_next_actions = np.argmax(next_Q_values, axis=1)

    
    
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards +
        (1 - dones) * discount_factor * next_best_Q_values)
    
    mask = tf.one_hot(actions, n_outputs)
# --------------------- Double DQN --------------------- #

    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info, skoupidi = env.step(action)  # one extra variable 
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


# Buiilding a double DQN model

# Create the online model 
model = build_model(2)

# Create the target model
target = keras.models.clone_model(model)

# Copy the weights from the online model to the target model
target.set_weights(model.get_weights())

# model training
eps = 300
ep_rewards = []
for episode in range(eps):
    obs, info = env.reset()

    cumulative_reward = 0

    ### one episode
    for step in range(475):
        epsilon = max(1 - episode / 500, 0.02)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        cumulative_reward += reward
        if done:
            break
    ###

    if episode > 50:  # and episode < int(2*eps/3)
        training_step(batch_size, target)
        if episode % target_update_freq == 0:
            target.set_weights(model.get_weights())
    ep_rewards += [cumulative_reward]


# print the average reward over the last 100 episodes
print("Average reward over the last 100 ep",np.mean(ep_rewards[-100:]))
