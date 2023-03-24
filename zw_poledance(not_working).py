import numpy as np
import tensorflow as tf
import gymnasium as gym
import sys
from collections import deque

from Helper import make_tensor, stable_loss, softmax, e_greedy, linear_anneal


args = sys.argv[1:]

# while still debugging, always keep target network and exp replay on
debug = True

if debug:
    print('DEBUG')
    tn_active = True
    er_active = True

# handling of command line arguments
for arg in args:
    if arg == '--target_network':
        tn_active = True
    elif arg == '--experience_replay':
        er_active = True
    try:
        max_eps = int(arg)
        print('max ep count:', max_eps)
    except:
        continue


class Q_Network():
    '''General deep Q Network'''

    def __init__(self, learning_rate: float, optimizer: str, batch_size: int = 32 ):
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)  # change this!
        self.batch_size = batch_size
        a1, a2, a3, a4 = 0,0,0,1
        self.gamma = 0.99

        if a1:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(20, activation='relu', kernel_initializer='he_uniform', input_shape=(4,)),
                tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.Dense(2, activation='linear')
            ])
        elif a2:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(30, activation='relu', kernel_initializer='he_uniform', input_shape=(4,)),
                tf.keras.layers.Dense(2, activation='linear')
            ])
        elif a3:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_uniform', input_shape=(4,)),
                tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_uniform'),
                tf.keras.layers.Dense(2, activation='linear')
            ])
        elif a4:  # book architecture
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='elu', input_shape=(4,)),
                tf.keras.layers.Dense(32, activation='elu'),
                tf.keras.layers.Dense(2, activation='linear')
            ])

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        #self.model.compile(optimizer, loss=stable_loss)

    def update(self, states, actions, rewards, dones, target_output):
        '''learning of network'''
        states = make_tensor(states, list=True)
        # target value to compare to np.ones((len(dones)))
        print(1-dones)
        print()
        print(target_output)
        target_val = rewards + (1-dones) * self.gamma * np.max(target_output, axis=1)
        #print(target_val)
        mask = tf.one_hot(actions, 2)

        with tf.GradientTape() as tape:
            Q_vals = self.model(states)
            Q_val = tf.reduce_sum(Q_vals * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(stable_loss(target_val, Q_val))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class DQN_Agent():
    '''Class for the learning net'''

    def __init__(self, batch_size: int = 32, replay: bool = True, epsilon: float = 0, temp: float = 0, max_buffer: int = int(1e3) ):
        self.epsilon = epsilon
        self.temp = temp
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.replay = replay
        self.replay_buffer = deque(maxlen=2000)

        self.states, self.actions, self.rewards, self.n_states, self.dones = [], [], [] , [], []
        self.sample = []  # init not really needed, just for overview of attributes

    def draw_action(self, s, q_network: Q_Network):
        '''
        input: 
        - s: (4dim) state of environment
        returns:
        - int 0 or 1, action according to QNet and policy
        no learning is happening here '''

        s_tensor = make_tensor(s, False)
        Q_vals = q_network.model.predict(s_tensor, verbose=0)  # outputs two Q-values

        if self.epsilon:
            return e_greedy(Q_vals, epsilon=self.epsilon)
        elif self.temp:
            return softmax(Q_vals, temp=self.temp)
        else:
            print('No policy given! Default to greedy!')
            return np.argmax(Q_vals)

    def buffer_clip(self, times):
        '''reduce buffer length when exceeding memory'''
        for _ in range(times):
            d = np.random.randint(len(self.buffer))
            self.buffer.pop(d)

    def draw_sample(self):
        '''create (random) sample of length batch_size to be trained with'''
        '''
        if len(self.buffer[0]) <= self.batch_size:
            states, actions, rewards, next_states, dones = self.buffer

        elif self.replay:
            choices = np.random.randint(len(self.buffer[0]), size=self.batch_size)
            batch = [[self.buffer[i][choice] for choice in choices] for i in range(5)]
            states, actions, rewards, next_states, dones = batch
        else:
            batch = [self.buffer[i][-self.batch_size:] for i in range(5)]
            states, actions, rewards, next_states, dones = batch
        return states, actions, rewards, next_states, dones
        '''
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size) 
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] 
                                                                  for experience in batch]) 
                                                                  for field_index in range(5)]
        return states, actions, rewards, next_states, dones


    def buffer_update(self, state, action, reward, next_state, done):
            ''' append newly obtained environment state to memory'''
            self.states += [state] 
            self.actions += [action]
            self.rewards += [reward]
            self.n_states += [next_state]
            self.dones += [done]
            self.buffer = [self.states, self.actions, self.rewards, self.n_states, self.dones]
            self.replay_buffer.append((self.states, self.actions, self.rewards, self.n_states, self.dones))
            '''
            if len(self.replay_buffer) > self.max_buffer:
                self.replay_buffer[-self.max_buffer:]
            '''

def q_learning(max_eps: int, learning_rate: float = 0.001, epsilon: float = None, temp: float = None, 
               optimizer: str = 'rmsprop', batch_size: int = 32, update_target_freq: int = 100,
               tn_active: bool =tn_active, er_active: bool=er_active, 
               budget: int = 10000, run: str = '1', save: bool = False):
    '''
    Method that trains a DQN for a set amount of episodes, returns array of cumulative rewards of each episode (length of episode)
    - max_eps: the number of episodes to play
    - learning_rate: learning rate
    - epsilon: (annealing) epsilon-greedy parameter
    - temp: temperature of Boltzmann policy
    - optimizer: optimizer architecture of the Q Network
    - batch_size: amount of (random) samples used to train the Q Network
    - update_target_freq: amount of steps that pass between each update of target network (when active)
    - tn_active, er_active: bool of whether to include target network or experience replay
    - budget: max amount of steps we allow a session (many episodes) to take
    - run: string for identification of test (not used)
    - save: exporting of intermediate data (not used)
    '''

    if tn_active:
        print()
        print('Activating target network...')
    if er_active:
        print('Activating experience replay...')
        print()

    max_episode_length = 500  # CartPole limits naturally at 475
    train_model_freq = 4

    # init game
    env = gym.make("CartPole-v1")  # , render_mode='human'

    q_network = Q_Network(learning_rate, optimizer, batch_size=batch_size)
    target_network = Q_Network(learning_rate, optimizer, batch_size=batch_size)
    target_network.model.set_weights(q_network.model.get_weights())

    agent = DQN_Agent(epsilon=epsilon, batch_size=batch_size)
    total_rewards = []  # collects cumulative reward for each episode
    step_count = 0  # counts interactions with the environment
    ep_count = 0  # counts terminated/truncated episodes


    while step_count < budget:  # to limit environment interaction
        agent.epsilon = linear_anneal(ep_count, max_eps, epsilon, 0.2*epsilon, 0.7)
        ep_count += 1
        ep_reward = 0
        timestep = 0

        if ep_count % 20 == 0 and debug:
            print(f"mean reward of last 20 {np.mean(total_rewards[-20:])}")

        state, info = env.reset()

        while timestep < max_episode_length:
            timestep += 1  # counts episode length
            step_count += 1  # counts overall interactions with env

            action = agent.draw_action(state, q_network)
            next_state, reward, term, trunk, info = env.step(action=action)
            
            agent.buffer_update(state, action, reward, next_state, term or trunk)
            ep_reward += reward
            state = next_state

            if ep_count > 20 and timestep > 32:
                # training Q net
                states, actions, rewards, next_states, dones = agent.draw_sample()
                print(np.array(next_states).shape)
                target_output = target_network.model.predict(next_states, verbose=0)
                q_network.update(states, actions, rewards, dones, target_output)

                if not tn_active:
                    # keeping target same as Q net
                    target_network.model.set_weights(q_network.model.get_weights())

            if step_count % update_target_freq == 0 and tn_active:
                # only update target every n-th step
                target_network.model.set_weights(q_network.model.get_weights())

            if term or trunk:
                total_rewards += [ep_reward]
                state, info = env.reset()
                break

        if not term and not trunk:  # equivalent to saying timestep >= max_episode_length
            total_rewards += [ep_reward]

        if ep_count % 100 == 0 and save:
            np.save(f'runs/rewards{run}', np.array(total_rewards))
        if ep_count >= max_eps:
            break

    env.close()
    return total_rewards

if __name__ == "__main__":
    rewards = q_learning(max_eps=300, learning_rate=0.001, epsilon=0.5, tn_active=tn_active, er_active=er_active, save=True)
    print(rewards)
