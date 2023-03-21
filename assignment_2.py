import numpy as np
from tensorflow import keras
import tensorflow as tf
import gymnasium as gym
import sys

from Helper import make_tensor, stable_loss, softmax, e_greedy

# constants / initializations
nb_actions = 2
losses = []
done = False
budget = 1e5
ep_count = 0  # counts finished episodes
ep_reward = 0
step_count = 0  # counts interactions with the environment

args = sys.argv[1:]
# while still debugging, always keep target network and exp replay on
debug = True
if debug:
    print('DEBUG')
    args = ['--target_network', '--experience_replay']


# init game
env = gym.make("CartPole-v1")  # , render_mode='human'
state, info = env.reset()


# hyperparameters
learning_rate = 0.01
epsilon = 0.1
temp = 1.0
max_buffer_length = int(1e4)
train_model_freq = 4
# must not be too high or else it will never be updated
update_target_freq = int(1000)
max_episode_length = int(1000)
batch_size = 32
gamma = 0.99
output_activation = None


class DQN_Agent():
    '''Class for the learning net'''

    def __init__(self, learning_rate: float, target_active: bool = True, replay: bool = True,
                 batch_size: int = 32, epsilon: float = 0, temp: float = 0, args=None):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.temp = temp

        self.target_active = '--target_network' in args
        self.replay = '--experience_replay' in args

        if self.target_active:
            print('Activating target network...')
        if self.replay:
            print('Activating experience replay...')

        self.batch_size = batch_size

        self.state_buffer = []
        self.reward_buffer = []
        self.big_R = []

        self.q_net = tf.keras.Sequential([
            keras.layers.Dense(
                20, activation='relu', kernel_initializer='glorot_uniform', input_shape=(4,)),
            keras.layers.Dense(10, activation='relu',
                               kernel_initializer='glorot_uniform'),
            keras.layers.Dense(nb_actions, activation=output_activation)
        ])

        self.q_net.summary()
        self.target_net = self.q_net
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.q_net.compile(self.optimizer, loss=stable_loss)

    def draw_action(self, s):
        '''
        input: 
        - s: (4dim) state of environment
        returns:
        - int 0 or 1, action according to QNet and policy
        no learning is happening here '''
        # print(f'type of s {type(s)}')
        s_tensor = make_tensor(s, False)
        Q_vals = self.q_net.predict(s_tensor, verbose=0)

        if self.epsilon:
            return e_greedy(Q_vals, epsilon=self.epsilon)
        elif self.temp:
            return softmax(Q_vals, temp=self.temp)
        else:
            print('No policy given! Default to greedy!')
            return np.random.randint(0, 2)

    def draw_sample(self):
        '''create random sample of length batch_size to be trained with'''
        if len(self.state_buffer) <= self.batch_size:
            self.state_sample = np.array(self.state_buffer)
            self.reward_sample = np.array(self.reward_buffer)
            return
        if self.replay:
            choice = np.random.choice(np.arange(len(self.state_buffer)), size=(
                self.batch_size,), replace=False)
            self.state_sample = np.array(self.state_buffer)[choice]
            self.reward_sample = np.array(self.reward_buffer)[choice]
        else:  # w/out experience we still want to batch the last 32 samples?
            self.state_sample = self.state_buffer[-self.batch_size:]
            self.reward_sample = self.reward_buffer[-self.batch_size:]

    def update(self):
        '''learning of behavioural network'''
        states = make_tensor(self.state_sample, list=True)
        target_output = self.target_net.predict(states, verbose=0)
        target_val = self.reward_sample + gamma * \
            np.max(target_output, axis=1)  # target value to compare to

        history = self.q_net.fit(
            states, target_val, batch_size=batch_size, verbose=0)
        loss = history.history['loss'][0]

        if not self.target_active:  # turn on/off target network
            self.target_net = self.q_net
            print('target equals qnet')
        return loss

    def target_update(self):
        ''' update target network to current QNets weights'''
        self.target_net = self.q_net
        return self.target_net

    def buffer_clip(self, times):
        '''reduce buffer length when exceeding memory'''
        for _ in range(times):
            d = np.random.randint(len(self.state_buffer))
            self.reward_buffer.pop(d)
            self.state_buffer.pop(d)

    def buffer_update(self, state, reward):
        ''' append newly obtained environment state and reward to memory'''
        self.state_buffer.append(state)
        self.reward_buffer.append(reward)


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):

    agent = DQN_Agent(learning_rate=learning_rate,
                      epsilon=epsilon, batch_size=batch_size, args=args)
    ep_count = 0
    total_rewards = []
    loss = 0
    while True:
        ep_count += 1
        ep_reward = 0
        if ep_count % 20 == 0:
            print(f"mean reward of last 20 {np.mean(agent.big_R[-20:])}")
        timestep = 0

        state, info = env.reset()

        while timestep < max_episode_length:
            timestep += 1
            step_count += 1

            action = agent.draw_action(state)
            next_state, r, term, trunk, info = env.step(action=action)
            agent.buffer_update(state, r)
            ep_reward += r

            if len(agent.state_buffer) > max_buffer_length:
                agent.buffer_clip(1)

            state = next_state

            # sample buffer
            agent.draw_sample()

            if step_count % train_model_freq == 0:
                loss = agent.update()

            if step_count % update_target_freq == 0 and agent.target_active:
                agent.target_update()
                # important to see, how often target is updated
                print('target update!')

            if term or trunk:
                agent.big_R += [ep_reward]
                state, info = env.reset()
                break
        # end of for loop
        total_rewards.append(ep_reward)
        losses += [loss]  # append episode's last loss to export later

        if ep_count >= 200:
            np.save('runs/all_ep_rewards', np.array(agent.big_R))
            np.save('runs/all_losses', np.array(losses))
            print(f'all rewards {agent.big_R}')
            break

    return total_rewards
