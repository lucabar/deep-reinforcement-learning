# write here a program that loads existing weights and plays without learning
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque # for experience replay
import h5py

### ideas

loss_fn = keras.losses.mean_squared_error
env = gym.make("CartPole-v1")


def build_model(j: int = 1, activ: str = "relu", init: str = "glorot_uniform"):
    # init = "he_normal"  # <--- here?
    input_shape = [4] # == env.observation_space.shape
    n_outputs = 2 # == env.action_space.n
    if j == 1:
        model = keras.models.Sequential([
        keras.layers.Dense(64, activation=activ, input_shape=input_shape,kernel_initializer=init),
        keras.layers.Dense(n_outputs)
        ])
    elif j == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation=activ, input_shape=(4,),kernel_initializer=init),
            tf.keras.layers.Dense(2)
        ])
    elif j == 3:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activ, input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(32, activation=activ, kernel_initializer=init),
            tf.keras.layers.Dense(2)
        ])
    elif j == 4:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activ, input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(64, activation=activ, kernel_initializer=init),
            tf.keras.layers.Dense(2)
        ])
    model.summary()
    return model

def build(arch: int = 1, lr: float = 0.0001, exist_model: str = None):
    model = build_model(arch)
    if exist_model:
        model.load_weights(exist_model)
        optimizer = None
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
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
    next_Q_values = target.predict(next_states, verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                        (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, 2)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return Q_values

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

## hyperparameters
learning_rate, batch_size, arch, target_update_freq, replay_buffer_size = (0.0001, 32, 4, 10, 5000)

discount_factor = 0.95
# iterate over all hyperparameters
count = 0

print()
start = time.time()
stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))
outp = f"---Starting Test {stamp}---\nparams:{learning_rate, batch_size, arch, target_update_freq, replay_buffer_size}"
print(outp)

# build model
replay_buffer = deque(maxlen=replay_buffer_size)

##### insert weight path
exist_weights = None  # activate when you want to learn
# exist_weights = "try_next_w_30_164938.h5"  # path to existing weights
# exist_weights = "runs/book/weights/" + exist_weights
#####

model, optimizer, target = build(arch, learning_rate, exist_weights)
ep_rewards = []
eps = 500  # <--- here?
max_mean = 100

try:
    for episode in range(eps):
        obs = env.reset()
        if optimizer:
            epsilon = max(1 - episode / 500, 0.02)
            #epsilon = max(1 - np.mean(ep_rewards)/200, 0.01)  # idea: couple annealing epsilon not to ep count but reward?
        else:  # not learning, not exploring, just greedy
            epsilon = 0.0

        cumulative_reward = 0

        ### one episode
        for step in range(475):
            obs, reward, done, info = play_one_step(env, obs, epsilon)
            cumulative_reward += reward
            if done:
                break
            # model training when no existing weight path is given
            if episode > 50 and optimizer:
                if episode == 51:
                    prnt = "We're now learning..."
                    outp += "\n"+ prnt
                    print(prnt)
                Q = training_step(batch_size, target)
                if episode % target_update_freq == 0:
                    target.set_weights(model.get_weights())
            else:
                if episode == 51:
                    prnt = "Not learning, but playing..."
                    outp += "\n"+ prnt
                    print(prnt)
        ### end of episode

        ep_rewards += [cumulative_reward]
        try:
            mean = round(np.mean(ep_rewards[-100:]),3)
        except:
            mean = round(np.mean(ep_rewards),3)
        if episode % 100 == 0 and episode > 50:
            prnt = f"Average of last 100: {mean}"
            outp += prnt
            print(f"Average of last 100: {mean}")
        if episode % 20 == 0:
            np.save(f"runs/book/rew_{stamp}.npy",np.array(ep_rewards))

        if  mean > max_mean and optimizer:
            # run without learning
            max_mean = mean
            prnt = "again saving weights"
            print(prnt)
            outp += "\n"+prnt
            model.save_weights(f"runs/book/weights/w_{stamp}.h5", overwrite=True)

    # save the best hyperparameters
    rew_mean = round(np.mean(ep_rewards[50:]),3)
    rew_median = np.median(ep_rewards[50:], axis=0)

    ticks = round((time.time()-start)/60,2)
    prnt = f"It took {ticks}mins."
    outp += "\n"+ prnt +"\n"
    print(prnt)

    prnt= f"\nMean reward:{rew_mean}, median: {rew_median}\nTime: {ticks}mins\n\n"
    print(prnt)
    outp += prnt

except:
    print("something went wrong")

finally:
    np.save(f"runs/book/rew_{stamp}.npy",np.array(ep_rewards))
    with open("runs/book/results/documentation.txt", 'a') as f:
        # export comand line output for later investigation
        f.write("\nFAILED!!\n"+outp)
