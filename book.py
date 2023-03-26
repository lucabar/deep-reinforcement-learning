import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from collections import deque # for experience replay

replay_buffer = deque(maxlen=2000) # 2000 is the maximum number of transitions we want to store

# constants / initializations
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.mean_squared_error
total_rewards = []

env = gym.make("CartPole-v1")

input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

def build():
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.mean_squared_error
    model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
    ])
    model.compile(optimizer, loss_fn)
    return model, optimizer, replay_buffer

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        
        # make the state a tensor --> the problem is that the state some specific times comes as a tuple and not as a numpy array !!!!!!
        
        #state = tf.convert_to_tensor(state) try 1
        #state = tf.ragged.constant(state) try 2
        # print( 'state: ', state, 'type: ', type(state)) # , 'dtype: ', state.dtype --> tuples dont have dtype

        # if type(state) == tuple:
            # print('It was tuple')
            #state = np.asarray(state).astype(np.float32)
            #state = np.array(state)
            #state = np.asanyarray(state)
            #array_state = np.array([])
            #for i in range(len(state)):
            #    array_state = np.append(array_state, state[i])
            #print( 'Now for the state: ', array_state, ' the type changed to: ', type(array_state))
            #state = array_state.astype(np.float32)
            # state = np.array(state)
            Q_values = model.predict(state[np.newaxis],verbose=0) # outputs two Q-values [np.newaxis, :]
            return np.argmax(Q_values[0]) # [0]


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states,verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                        (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        # print('all_Q_values: ', all_Q_values)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play_one_step(env, state, epsilon):
    
    action = epsilon_greedy_policy(state, epsilon)
    
    next_state, reward, done, info = env.step(action)
    # print('next_state: ', next_state, 'type: ', type(next_state))
    #position, velocity, angle, angular_vel = env.observation
    # !!!!!!!!!!!!!!!!!!!!!
    #next_state, reward, term, trunk, done, info = env.step(action=action)    

    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

n = 1
eps = 100
total_rewards=np.empty([n, eps])

for run in range(n):
    print(f'--run {run}--')
    model, optimizer, replay_buffer = build()
    ep_rewards = []
    for episode in range(eps):
        obs = env.reset()
        if episode % 20 == 0:
            print('ep: ', episode)
        R = 0
        for step in range(475):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = play_one_step(env, obs, epsilon)
            R += reward
            if done:
                break

        if episode > 50:
            training_step(batch_size)
        ep_rewards += [R]
    total_rewards[run] = ep_rewards
    plt.plot(ep_rewards)
    plt.savefig('temp_rew.pdf')


total_rewards = np.array(total_rewards)
total_rewards = np.mean(total_rewards, axis=0)
plt.plot(total_rewards)
plt.show()
