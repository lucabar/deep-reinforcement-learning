import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque  # for experience replay
from Helper import make_tensor, e_greedy, softmax, linear_anneal

# ideas
# hyperparameters
# default:2000 is the maximum number of transitions we want to store
replay_buffer = deque(maxlen=4000)

# architecture
# try only one layer

# constants / initializations
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error
total_rewards = []
target_update_freq = 10

env = gym.make("CartPole-v1")


input_shape = [4]  # == env.observation_space.shape
n_outputs = 2  # == env.action_space.n


def build_model(j: int = 1, activ: str = "elu", init: str = None):
    init = "he_normal"  # <--- here?
    if j == 1:
        model = keras.models.Sequential([
            # use 64 neurons in only one layer
            # use 512, 256, 64
            keras.layers.Dense(
                64, activation=activ, input_shape=input_shape, kernel_initializer=init),
            # keras.layers.Dense(32, activation="elu"),
            keras.layers.Dense(n_outputs)
            # tune (output) activation relu or tanh maybe, output linear
        ])
    elif j == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=activ,
                                  input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(2)
        ])
    elif j == 3:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=activ,
                                  input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(32, activation=activ,
                                  kernel_initializer=init),
            tf.keras.layers.Dense(2)
        ])
    elif j == 4:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation=activ,
                                  input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(16, activation=activ
                                  ),
            tf.keras.layers.Dense(16, activation=activ
                                  ),
            tf.keras.layers.Dense(2)
        ])
    elif j == 'Dueling':
        K = keras.backend
        
        input_states = keras.layers.Input(shape=input_shape)
        hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
        hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)

        # This creates a fully connected layer with a single neuron and no activation function, which will output the estimated state value for the input state.
        state_values = keras.layers.Dense(1)(hidden2)
        
        # This creates another fully connected layer with n_outputs neurons and no activation function, which will output the raw advantage estimates for each action.
        raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
        
        # This calculates the advantages by subtracting the maximum advantage estimate from each estimate, which helps to stabilize the learning process.
        advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
        
        # This combines the state values and advantages to compute the Q-values for each action.
        Q_values = state_values + advantages

        # now the model has two outputs, one for the state values and one for the advantages. 
        model = keras.Model(inputs=[input_states], outputs=[Q_values])
    return model


def build(j: int = 1, lr: float = 0.01):
    # 1 decrease replay
    # 2 change learning rate 0.05
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
        # outputs two Q-values [np.newaxis, :]
        Q_values = model.predict(state[np.newaxis], verbose=0)
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
    return Q_values


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, trunc, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done or trunc))
    return next_state, reward, done, trunc, info

# hyperparameter testing


learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64]
model_archs = [1, 2, 3]
target_update_freqs = [10, 100, 500]
replay_buffer_sizes = [1000, 3000, 5000]

best_hyperparameters = None
gold_reward = 0
silver_reward = 0
bronze_reward = 0

gold_median = 0

# iterate over all hyperparameters
count = 1

print("hyperparams: learning rate, batch_size, model_arch, target_update_freq, replay_buffer_size")

lr, batch_size, model_arch, target_update_freq, replay_buffer_size = (
    0.0001, 32, 1, 10, 5000)


start = time.time()
print(
    f"---Starting Test {count}, {lr, batch_size,model_arch, target_update_freq, replay_buffer_size}---")

# build model
replay_buffer = deque(maxlen=replay_buffer_size)
model, optimizer, target = build(model_arch, lr)
ep_rewards = []
eps = 500  # <--- here?
epsilon_initial = 0.4

epsilon = epsilon_initial
# model training
for episode in range(eps):
    obs, info = env.reset()
    # idea: couple annealing epsilon not to ep count but reward?
    # epsilon = max(1 - episode / 500, 0.02)
    # epsilon = max(1 - np.mean(ep_rewards)/200, epsilon)  # <--- here?
    if episode > 100:
        epsilon = linear_anneal(episode, eps, epsilon_initial, 0.01, 1)

    cumulative_reward = 0

    # one episode
    for step in range(475):
        obs, reward, done, trunc, info = play_one_step(env, obs, epsilon)
        cumulative_reward += reward
        if done or trunc:
            break
    ###

    if episode > 50:  # and episode < int(2*eps/3)
        Q = training_step(batch_size, target)
        if episode % target_update_freq == 0:
            target.set_weights(model.get_weights())
        if episode % 50 == 0:
            print(f"mean of last 50: {round(np.mean(ep_rewards[-50:]),3)}\n")
    ep_rewards += [cumulative_reward]

# save the best hyperparameters
rew_mean = round(np.mean(ep_rewards[50:]), 3)
rew_median = np.median(ep_rewards[50:], axis=0)
if rew_mean > gold_reward:
    gold_hyperparameters = (lr, model_arch, batch_size,
                            target_update_freq, replay_buffer_size)
    gold_reward = rew_mean
if rew_median > gold_median:
    gold_median = rew_median
    gold_med_hyperparam = (lr, model_arch, batch_size,
                           target_update_freq, replay_buffer_size)
if rew_mean > 100:
    model.save_weights(
        f"runs/book/weights/weights_{count}_rew{rew_mean}.h5", overwrite=True)

print(f"Best reward {gold_reward}, highest median {gold_median}")
ticks = round((time.time()-start)/60, 2)
print(f"It took {ticks}mins.")
print()
np.save(f"runs/book/rew{count}.npy", np.array(ep_rewards))
save = f"Test: {count}, params: {lr, batch_size,model_arch, target_update_freq, replay_buffer_size}"
save += f"\nMean reward:{rew_mean}, median: {rew_median}\nTime: {ticks}mins\n\n"
with open("runs/book/results/documentation.txt", 'a') as f:
    f.write(save)

print('best hyperparameters: ', gold_hyperparameters)
print('best avg reward: ', gold_reward)
print('best median: ', gold_median)
print('best hyperparams (med): ', gold_med_hyperparam)
