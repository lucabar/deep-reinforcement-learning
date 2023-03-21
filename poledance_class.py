import numpy as np
from tensorflow import keras
import tensorflow as tf
import gymnasium as gym
import sys


'''
TO DO:
    -
'''
# constants / initializations
nb_actions = 2
losses = []
done = False
budget = 1e5
ep_count = 0  # counts finished episodes
ep_reward = 0
max_ep_count = 500
step_count = 0  # counts interactions with the environment

args = sys.argv[1:]
# while still debugging, always keep target network and exp replay on
debug = True
train = True
for arg in args:
    try:
        max_ep_count = int(arg)
        print('max ep count:', max_ep_count)
    except:
        continue

if debug:
    print('DEBUG')
    args = ['--experience_replay']

# init game
env = gym.make("CartPole-v1")  # , render_mode='human'
state, info = env.reset()
# print(observation)  # 4 state values define one state


# hyperparameters
learning_rate = 0.001
epsilon = 0.1
temp = 1.0
max_buffer_length = int(1e4)
train_model_freq = 4
update_target_freq = int(1000)  # must not be too high or else it will never be updated
max_episode_length = int(1000)
batch_size = 32
gamma = 0.99
output_activation = None

## helping functions

def make_tensor(s, list: bool):
    '''in order to be used in net.predict() method'''
    s_tensor = tf.convert_to_tensor(s)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)

def stable_loss(target, pred):  # implement own loss on stable target
    '''Squared loss'''
    squared_difference = tf.square(target - pred)
    return tf.keras.losses.MeanSquaredError(target,pred)  # Note the `axis=-1`

def softmax(Q_vals, temp):
    '''copied from Assignment 1 Helper'''
    Q_vals = Q_vals / temp # scale by temperature
    z = Q_vals - max(Q_vals) # substract max to prevent overflow of softmax 
    probs = np.exp(z)/np.sum(np.exp(z)) # compute softmax returns prob distrib
    return np.random.choice(2, None, p=probs)

def e_greedy(Q_vals, epsilon):
    ''' epsilon greedy policy '''
    if np.random.uniform(0.,1) > epsilon:
            return np.argmax(Q_vals)
    else:
        return np.random.randint(0,2)

#######

class DQN_Agent():
    '''Class for the learning net'''
    def __init__(self, learning_rate: float, target_active: bool = True, replay: bool = True, 
                 batch_size: int = 32, epsilon: float = 0, temp: float = 0, args = None):
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
        keras.layers.Dense(20, activation='relu',kernel_initializer='he_uniform',input_shape=(4,)),
        keras.layers.Dense(10, activation='relu',kernel_initializer='he_uniform'),
        keras.layers.Dense(nb_actions, activation=output_activation)
    ])

        self.q_net.summary()
        self.target_net = self.q_net
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.q_net.compile(self.optimizer,loss="mean_squared_error") # stable_loss
        self.target_net.compile(self.optimizer,loss="mean_squared_error")

    def draw_action(self, s):
        '''
        input: 
        - s: (4dim) state of environment
        returns:
        - int 0 or 1, action according to QNet and policy
        no learning is happening here '''
        #print(f'type of s {type(s)}')
        s_tensor = make_tensor(s, False)
        self.output = self.q_net.predict(s_tensor, verbose=0)

        if self.epsilon:
            return e_greedy(self.output, epsilon=self.epsilon)
        elif self.temp:
            return softmax(self.output, temp=self.temp)
        else:
            print('No policy given! Default to greedy!')
            return np.random.randint(0,2)


    def draw_sample(self):
        '''create random sample of length batch_size to be trained with'''
        #print(f"buffer size: {len(self.state_buffer)}")
        if len(self.state_buffer) <= self.batch_size:
                self.state_sample = np.array(self.state_buffer)
                self.reward_sample = np.array(self.reward_buffer)
                return
        if self.replay:
            choice = np.random.choice(np.arange(len(self.state_buffer)), size = (self.batch_size,), replace=False)
            # print('random choice head ', choice[:5])
            self.state_sample = np.array(self.state_buffer)[choice]
            self.reward_sample = np.array(self.reward_buffer)[choice]
            #print('samples drawn: ', self.state_sample[:5])

        else:  # w/out experience we still want to batch the last 32 samples?
            self.state_sample = self.state_buffer[-self.batch_size:]
            self.reward_sample = self.reward_buffer[-self.batch_size:]

    
    def update(self):
        '''learning of behavioural network'''
        states = make_tensor(self.state_sample, list=True)
        target_output = self.target_net.predict(states, verbose=0)
        target_val = self.reward_sample + gamma *  np.max(target_output, axis=1)  # target value to compare to
        ''' THIS DEFINITELY STILL NEEDS WORK. As soon as the target update is done, it stops learning (I think)'''
        history = self.q_net.fit(states,target_val,batch_size=batch_size, verbose=0)
        loss = history.history['loss'][0]
        
        if not self.target_active:  # turn on/off target network 
            self.target_net = self.q_net
            # print('target equals qnet')
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


agent = DQN_Agent(learning_rate=learning_rate, epsilon=epsilon, batch_size=batch_size, args=args)

while True:
    ep_count += 1
    ep_reward = 0
    if ep_count % 20 == 0:
        print(f"mean reward of last 20 {np.mean(agent.big_R[-20:])}")
    timestep = 0

    while timestep < max_episode_length:
        timestep += 1
        step_count += 1
        # draw action
        action = agent.draw_action(state)
        next_state, r, term, trunk, info = env.step(action=action)
        agent.buffer_update(state, r)
        ep_reward += r

        if len(agent.state_buffer) > max_buffer_length:
            agent.buffer_clip(1)

        state = next_state

        # sample buffer
        agent.draw_sample()

        if step_count % train_model_freq == 0 and train:
            loss = agent.update()
            losses += [loss]
        if step_count % update_target_freq == 0 and agent.target_active and train:
            agent.target_update()
            print('target update!')  # important to see, how often target is updated

        if term or trunk:
            agent.big_R += [ep_reward]
            state, info = env.reset()
            break
    # end of for loop

      # append episode's last loss to export later
    if ep_count % 100 == 0:
        np.save(f'runs/all_ep_rewards', np.array(agent.big_R))
        np.save(f'runs/all_losses', np.array(losses))

    if ep_count >= max_ep_count:
        break
