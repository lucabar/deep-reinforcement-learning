import numpy as np
#from tf_agents import agents
from tensorflow import keras
import tensorflow as tf
import gymnasium as gym

'''
TO DO:
    - implement comand line arguments (no) experience_replay/target_network
'''
# constants / initializations
nb_actions = 2
buffer = []
counter = 0
reward = 0
done = False
budget = 1e5
ep_count = 0



# init game
env = gym.make("CartPole-v1")  # , render_mode='human'
state, info = env.reset()

# print(observation)  # 4 state values define one state


# hyperparameters
epsilon = 0.01
output_activation = None
learning_rate = 1e-3
max_buffer_length = 1e4
train_model_freq = 4
update_target_freq = 1e3
max_episode_length = 100

loss_func = keras.losses.Huber()




for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()


def stable_loss(target, pred):  # implement own loss on stable target
    squared_difference = tf.square(target - pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def init_Qnet():
    input = keras.layers.Input(shape=(4,))

    h1 = keras.layers.Dense(100, activation='relu',kernel_initializer='glorot_uniform')(input)
    h2 = keras.layers.Dense(50, activation='relu',kernel_initializer='glorot_uniform')(h1)
    output = keras.layers.Dense(nb_actions, activation=output_activation)(h2)
    return keras.Model(inputs=input, outputs=output)

def init_buffer():
    ''' initialize replay buffer (memory) '''
    return np.empty((1,4))

def draw_action(s, net, epsilon, greedy=True):
    '''
    input: 
    - observation: (4dim state) of environment
    - net: policy approximator network
    returns:
    - greedy action to act upon
    using pre-initialised -q_net-, predict an output. no training! '''

    s_tensor = tf.convert_to_tensor(s)
    s_tensor = tf.expand_dims(s_tensor, 0)
    action_probs = net(s_tensor, training=False)

    if greedy or np.random.uniform(0.,1) > epsilon:
        return np.argmax(action_probs)
    else:
        return np.random.randint(nb_actions)



q_net = init_Qnet()
target_q_net = q_net  # clone into target network
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

#q_net.summary()
#q_net.predict(observation)

while True:
    ep_reward = 0
    for timestep in range(1,max_episode_length):

        # draw action
        action = draw_action(state, q_net, epsilon = 0.9, greedy=False)
        next_state, r, term, trunk, info = env.step(action=action)

        ep_reward += r
        buffer.append([state, action, r, next_state])
        state = next_state


        done = term or trunk
        if done:
            break


