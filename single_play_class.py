import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from collections import deque # for experience replay
from Helper import make_tensor, e_greedy, softmax, linear_anneal
import matplotlib.pyplot as plt

class Q_Network():
    '''General deep Q Network'''
    def __init__(self, learning_rate: float = 0.0001, optimizer: str = 'adam', architecture: int = 4, path_to_weights: str = None, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.gamma = 0.99
        self.update_count = 0
        activ = "elu"
        init = "glorot_uniform"
        self.loss_fn = keras.losses.mean_squared_error
        self.path_to_weights = path_to_weights
        self.optimizer = optimizer
        self.batch_size = batch_size

        if architecture == 1:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation=activ, input_shape=(4,),kernel_initializer=init),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 2:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(30, activation=activ, kernel_initializer=init, input_shape=(4,)),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 3:
            self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activ, input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.Dense(32, activation=activ, kernel_initializer=init),
            tf.keras.layers.Dense(2)
            ])
        elif architecture == 4:
            self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activ, input_shape=(4,), kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation=activ, kernel_initializer=init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2)
            ])

        if path_to_weights:
            self.model.load_weights(path_to_weights)
        else:
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(self.optimizer, self.loss_fn)
        
        self.model.summary()
    
    def update(self, states, actions, rewards, dones, target_output):
        self.update_count += 1
        max_next_Q_values = np.max(target_output, axis=1)
        target_Q_values = (rewards +
                            (1 - dones) * self.gamma * max_next_Q_values)
        mask = tf.one_hot(actions, 2)
        
        self.model.fit(states, target_Q_values, batch_size=self.batch_size, verbose=0)
        
        # with tf.GradientTape() as tape:
        #     all_Q_values = self.model(states)
        #     Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        #     loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        # grads = tape.gradient(loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

class DQN_Agent():
    '''Class for the learning net'''
    def __init__(self, batch_size: int = 32, replay: bool = True, epsilon: float = 0, temp: float = 0, max_buffer: int = int(1e3), policy: str = 'epsilon' ):
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.temp = temp
        self.policy = policy
        self.batch_size = batch_size

        if replay:
            self.replay = replay
            self.buffer = deque(maxlen=max_buffer)

        self.states, self.actions, self.rewards, self.n_states, self.dones = [], [], [] , [], []
        self.sample = []  # init not really needed, just for overview of attributes

    def draw_action(self, state, q_network: Q_Network):
        '''
        input: 
        - s: (4dim) state of environment
        returns:
        - int 0 or 1, action according to QNet and policy
        no learning is happening here '''

        s_tensor = make_tensor(state, False)
        Q_vals = q_network.model.predict(s_tensor, verbose=0)  # outputs two Q-values

        if self.policy == 'epsilon':
            return e_greedy(Q_vals, epsilon=self.epsilon)
        elif self.policy == 'softmax':
            return softmax(Q_vals, temp=self.temp)
        else:
            print('No policy given! Default to greedy!')
            return np.argmax(Q_vals)

    def draw_sample(self):
        ''' append newly obtained environment state to memory'''
        indices = np.random.randint(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def buffer_update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


def learning(eps, learning_rate, batch_size, architecture, target_update_freq, replay_buffer_size, path_to_weights: str = None):
    start = time.time()
    stamp = time.strftime("%d_%H%M%S",time.gmtime(time.time()))
    outp = f"---Starting Test {stamp}---\nparams:{learning_rate, batch_size, arch, target_update_freq, replay_buffer_size}"
    print(outp)

    network = Q_Network(learning_rate,optimizer='adam',architecture=architecture,path_to_weights=path_to_weights)
    target = Q_Network()
    target.model = tf.keras.models.clone_model(network.model)
    target.model.set_weights(network.model.get_weights())
    agent = DQN_Agent(batch_size, replay=True, epsilon=1, max_buffer=replay_buffer_size)

    # start the environment
    env = gym.make("CartPole-v1")

    ep_rewards = []
    eps = 500
    max_mean = 100
    budget = 0


    for episode in range(eps):
        state = env.reset()
        # agent.epsilon = max(1 - episode / 500, 0.02)
        agent.epsilon = linear_anneal(episode, eps, agent.epsilon_initial, 0.01, 0.7)
        # epsilon = max(1 - np.mean(ep_rewards)/200, 0.01)  # idea: couple annealing epsilon not to ep count but reward?

        cumulative_reward = 0

        #### episode starts
        for step in range(475):
            budget += 1
            action = agent.draw_action(state, network)
            next_state, reward, done, info = env.step(action=action)

            agent.buffer_update(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward

            if done:
                break

        # model training when no existing weight path is given
        if episode > 50:
            if episode == 51:
                prnt = "We're now learning..."
                outp += "\n"+ prnt
                print(prnt)

            # do training
            states, actions, rewards, next_states, dones = agent.draw_sample()
            target_output = target.model.predict(next_states, verbose=0)
            network.update(states, actions, rewards, dones, target_output)

            # update target
            if network.update_count % target_update_freq == 0:
                target.model.set_weights(network.model.get_weights())

        elif not network.optimizer:
            if episode == 51:
                prnt = "Not learning, but playing..."
                outp += "\n"+ prnt
                print(prnt)


        #### episode is over

        ep_rewards += [cumulative_reward]

        try:
            mean = round(np.mean(ep_rewards[-50:]),3)
        except:
            mean = round(np.mean(ep_rewards),3)

        if episode % 50 == 0 and episode > 50:
            print(f"budget: {budget}")
            prnt = f"Average of last 50: {mean}"
            outp += prnt
            print(prnt)

        if episode % 20 == 0:
            np.save(f"runs/book/rew_{stamp}.npy",np.array(ep_rewards))

        if  mean > max_mean and network.path_to_weights:
            # run without learning
            max_mean = mean
            prnt = "again saving weights"
            print(prnt)
            outp += "\n"+prnt
            network.model.save_weights(f"runs/book/weights/w_{stamp}.h5", overwrite=True)

    #### all episodes are done
        # save the mean/median after episodes are finished
    rew_mean = round(np.mean(ep_rewards[50:]),3)
    rew_median = np.median(ep_rewards[50:], axis=0)

    ticks = round((time.time()-start)/60,2)
    prnt = f"It took {ticks}mins."
    outp += "\n"+ prnt +"\n"
    print(prnt)

    prnt= f"\nMean reward:{rew_mean}, median: {rew_median}\nTime: {ticks}mins\n\n"
    print(prnt)
    outp += prnt

    # except:
    #     print("something went wrong")

    # finally:
    #     np.save(f"runs/book/rew_{stamp}.npy",np.array(ep_rewards))
    with open("runs/book/results/documentation.txt", 'a') as f:
        # export comand line output for later investigation
        f.write("\nFAILED!!\n"+outp)
    return ep_rewards

if __name__ == "__main__":
    eps = 200
    n_runs = 5

    all_rewards = np.array((n_runs,eps))

    ## hyperparameters
    learning_rate, batch_size, arch, target_update_freq, replay_buffer_size = (0.0001, 32, 1, 10, 5000)

    for run in range(n_runs):
        rewards = learning(eps, learning_rate, batch_size, architecture=arch, target_update_freq=target_update_freq, replay_buffer_size=replay_buffer_size)
        all_rewards[run] = rewards

    plt.plot(np.mean(all_rewards,axis=0))
