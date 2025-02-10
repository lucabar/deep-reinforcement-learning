import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque # for experience replay

### ideas
## hyperparameters
replay_buffer = deque(maxlen=5000) # default:2000 is the maximum number of transitions we want to store 

## architecture 
# try only one layer

# constants / initializations
batch_size = 32
discount_factor = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss_fn = keras.losses.mean_squared_error
total_rewards = []
target_update_freq = 10


env = gym.make("CartPole-v1")


input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

def build_model(j: int = 1, activ: str = "elu"):
    if j == 1:
        # -------------------------- Dueling DQN --------------------------------

        # This imports the Keras backend, which provides access to low-level operations in TensorFlow.
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

# introduce Prioritized Experience Replay
def sample_experiences_prioritize(batch_size, beta=0.4):
    
    to_be_done = 'Here we need to sample the experiences based on the priorities'
    
    return 



def training_step(batch_size, target):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # --------------------- Dueling DQN --------------------- #
    # The concept of advantage is used to calculate the Q-values
    # In a Dueling DQN, the model estimates both the value of the state
    # and the advantage of each possible action.

    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    next_Q_values = target.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1, keepdims=True)
    target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)

  

    

    mask = tf.one_hot(actions, n_outputs)
 
   
    
# --------------------- Dueling DQN --------------------- #

    with tf.GradientTape() as tape:
        all_Q_values = model(states)  # should this also change to random selection???
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
model = build_model(1)

# Create the target model
target = keras.models.clone_model(model)

# Copy the weights from the online model to the target model
target.set_weights(model.get_weights())

# model training
eps = 600
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
    print("Episode: {}, Reward: {}, Epsilon: {}".format(episode, cumulative_reward, epsilon))


# print the average reward over the last 100 episodes
print("Average reward over the last 100 ep",np.mean(ep_rewards[-100:]))
