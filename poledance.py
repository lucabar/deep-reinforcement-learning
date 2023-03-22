import numpy as np
import tensorflow as tf
import gymnasium as gym
import sys

from Helper import make_tensor, stable_loss, softmax, e_greedy


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

    def __init__(self, learning_rate: float, optimizer: str, batch_size: int = 32, layers: list = [20,10] ):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.gamma = 0.99

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                20, activation='relu', kernel_initializer='he_uniform', input_shape=(4,)),
            tf.keras.layers.Dense(10, activation='relu',
                               kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer, loss=stable_loss)

    def update(self, states, target_output):
        '''learning of network'''
        rewards = np.ones(len(states))  # all rewards are 1
        states = make_tensor(states, list=True)
        # target value to compare to
        target_val = rewards + self.gamma * np.max(target_output, axis=1)

        self.model.fit(
            states, target_val, batch_size=self.batch_size, verbose=0)

    def get_model(self):
        return self.model


class DQN_Agent():
    '''Class for the learning net'''

    def __init__(self, batch_size: int = 32, replay: bool = True, epsilon: float = 0, temp: float = 0, max_buffer: int = int(1e4) ):
        self.epsilon = epsilon
        self.temp = temp
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.replay = replay

        self.state_buffer = []
        self.state_sample = []  # init not really needed, just for overview of attributes

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
            d = np.random.randint(len(self.state_buffer))
            self.state_buffer.pop(d)

    def draw_sample(self):
        '''create (random) sample of length batch_size to be trained with'''
        if len(self.state_buffer) <= self.batch_size:
            return np.array(self.state_buffer)
        if self.replay:
            choice = np.random.choice(np.arange(len(self.state_buffer)), size=(self.batch_size,))  # add replace=False ?
            return np.array(self.state_buffer)[choice]
        else:  # w/out experience we still want to batch the last 32 samples?
            return self.state_buffer[-self.batch_size:]

    def buffer_update(self, state):
            ''' append newly obtained environment state to memory'''
            self.state_buffer.append(state)

            if len(self.state_buffer) > self.max_buffer:
                self.buffer_clip(1)

            self.state_sample = self.draw_sample()  # after having updated the buffer, update/create samples to learn from


def q_learning(max_eps: int, budget: int, learning_rate: float = 0.01, epsilon: float = 0.01, temp: float = None, optimizer: str = 'rmsprop',
               batch_size: int = 32, tn_active: bool =tn_active, er_active: bool=er_active, run: str = '1', save: bool = False):
    if tn_active:
        print('Activating target network...')
    if er_active:
        print('Activating experience replay...')

    max_episode_length = 500  # CartPole limits naturally at 475
    train_model_freq = 4
    update_target_freq = int(1000)


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
            next_state, r, term, trunk, info = env.step(action=action)
            agent.buffer_update(state)
            ep_reward += r

            state = next_state

            if step_count % train_model_freq == 0:
                # training Q net
                states = agent.state_sample
                target_output = target_network.model.predict(states, verbose=0)
                q_network.update(states, target_output)

                if not tn_active:
                    # keeping target same as Q net
                    target_network.model.set_weights(q_network.model.get_weights())


            if step_count % update_target_freq == 0 and tn_active:
                # only update target every n-th step
                target_network.model.set_weights(q_network.model.get_weights())
                # important to see, how often target is updated
                if debug:
                    print('target update!')

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

    return total_rewards

if __name__ == "__main__":
    budget = 100000
    max_eps = 200

    rewards = q_learning(max_eps, budget=budget, learning_rate=0.001, tn_active=tn_active, er_active=er_active, save=True)
