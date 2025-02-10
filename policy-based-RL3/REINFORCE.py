import numpy as np
from catch import Catch
import tensorflow as tf
from collections import deque
from Helper import time_it, print_it
import time


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']
seed = 42
rng = np.random.default_rng(seed=seed)

def make_tensor(state, list: bool = False):
    '''in order to be used in model.predict() method'''
    s_tensor = tf.convert_to_tensor(state)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)

class Actor():
    # @time_it
    def __init__(self, learning_rate: float = 0.0001, arch: int = 1, observation_type: str = "pixel", 
                 rows=7, columns=7, boot: str = "MC", n_step: int = 1, saved_weights: str = None,
                 seed=None, critic:bool=True):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.boot = boot
        self.n_step = n_step
        self.critic = critic
        self.gamma = 0.99
    
        # network parameters
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        # IMPLEMENT ENTROPY REGULARIZATION

        # gradient preparation
        # self.loss_fn = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        

        if (observation_type == 'pixel'):
            input_shape = (rows, columns, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)
        if arch == 1:
            input = tf.keras.layers.Input(shape=input_shape)
            dense = tf.keras.layers.Dense(
                64, activation=activ_func, kernel_initializer=init)(input)
            dense = tf.keras.layers.Dense(
                64, activation=activ_func, kernel_initializer=init)(dense)
            dense = tf.keras.layers.Flatten()(dense)


        if critic:
            output_value = tf.keras.layers.Dense(1, activation='linear')(dense)
            self.model = tf.keras.models.Model(inputs=input, outputs=[output_value])
        else:
            output_actions = tf.keras.layers.Dense(3, activation='softmax')(dense)
            self.model = tf.keras.models.Model(inputs=input, outputs=[output_actions])


        if saved_weights:
            print('## Working with pre-trained weights ##')
            self.model.load_weights(saved_weights)
        self.model.summary()

    def bootstrap(self, t, rewards, values = None):
        if self.boot == "MC":
            rewards = rewards[t:]
            return self.gamma**np.arange(0, len(rewards)) @ rewards

        elif self.boot == "n_step":
            rewards = rewards[t:t+self.n_step]
            return self.gamma**np.arange(0, self.n_step) @ rewards + self.gamma**self.n_step * values[t+self.n_step]

    def gain_fn(self, prob_out, Q, states, actions):
        if self.boot == 'MC':
            gain =  -Q * tf.math.log(prob_out)  # * -1 due to maximising instead of minimizing?
        elif self.boot == 'n_step':
            gain = - Q @ tf.math.log(prob_out)
        return gain

    def update_actor(self, state, action, Q):
        '''got code structure from https://keras.io/guides/writing_a_training_loop_from_scratch/'''
        action_n = np.where(ACTION_EFFECTS==action)[0][0]
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            if self.observation_type == "pixel":
                state = make_tensor(state.reshape(self.columns,self.rows,2))
            elif self.observation_type == "vector":
                state = make_tensor(state.reshape(3,))
            if self.boot == "MC":
                probs_out = self.model(state, training=True)
                value_out = None
            elif self.boot == "n_step":
                probs_out = self.model(state, training=True)
            # Compute the loss value for this minibatch.
            gain_value = self.gain_fn(tf.reshape(probs_out, [3])[action_n], Q)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(gain_value, self.model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return grads
        # self.optimizer.minimize(zgain_value, self.model.trainable_weights, tape=tape)


    def update_critic(self):
        pass

    def reshape_state(self, state):
        if self.observation_type == "pixel":
            return state.reshape(1,self.columns,self.rows,2)
        elif self.observation_type == "vector":
            return state.reshape(1,3)

@time_it
def reinforce(n_episodes:int=50, learning_rate:float=0.001, rows:int=7, columns:int=7, obs_type:str="pixel", max_misses:int=10, 
              max_steps:int=250, seed:int=None, speed:float=1.0, boot:str="MC", weights:str=None, minibatch:int=1, stamp:str=None):

    # IMPLEMENT (TENSORBOARD) CALLBACKS FOR ANALYZATION, long book 315

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps, max_misses=max_misses, observation_type=obs_type, seed=seed)

    if boot == "MC":
        n_step = max_steps
    elif boot == "n_step":
        n_step = 1  # or larger


    all_rewards = []
    actor = Actor(learning_rate, boot=boot, n_step=n_step, observation_type=obs_type, saved_weights=weights, seed=seed)
    if boot == 'n_step':
        critic = Actor(learning_rate, boot=boot, n_step=n_step, observation_type=obs_type, saved_weights=weights, seed=seed, critic=True)
    memory = deque(maxlen=max_steps)

    for ep in range(n_episodes):
        print()
        for m in range(minibatch):
            '''
            PROBLEM: For some reason after some episodes the output of the network
            is "nan" (the probabilities are non existent). Next step: figure this out
            suspect: learning rate highly important. error for lr >= 0.005
            '''
            ep_reward = 0
            memory.clear()
            state = actor.reshape_state(env.reset())
            # generate full trace
            for T in range(max_steps):
                if actor.boot == "MC":
                    action_p= actor.model.predict(state, verbose=0)
                    value = None
                elif actor.boot == "n_step":
                    action_p = actor.model.predict(state, verbose=0)
                try:
                    action = rng.choice(ACTION_EFFECTS, p=action_p.reshape(3,))
                    # print(f"good state: {state}")
                    # if T == 0:
                    #     print(f"good probabilities:",action_p)
                except:
                    print(f"faulty probabilities:",action_p)
                    break

                next_state, r, done = env.step(action)
                next_state = actor.reshape_state(next_state)
                ep_reward += r
                memory.append((state,action,r,next_state,done,value))

                if done:
                    break
                state = next_state

            # trace is finished
            print(f'{ep}, reward: {ep_reward}')
            all_rewards.append(ep_reward)

        # extract data from memory (to do: delete unused variables)
        memory = [memory[index] for index in range(len(memory))]
        states, actions, rewards, next_states, dones, values = [np.array([experience[field_index] 
                                                                    for experience in memory]) 
                                                                    for field_index in range(6)]
        
        Q_values = []
        # update loop
        for t in range(T):
            Q_values.append(actor.bootstrap(t,rewards, values))
            if actor.boot == 'MC':
                actor.update_actor(states[t], actions[t],Q_values[-1])
            elif actor.boot == 'n_step':
                actor.update_actor(states[t], actions[t],Q_values)  # in case V network is not updated separately!

        if ep % 10 == 0 and ep > 0:
            np.save(f'data/rewards/tmp_reward', all_rewards)
            
    actor.model.save_weights(f'data/weights/w_{stamp}.h5')
    np.save(f'data/rewards/r_{stamp}', all_rewards)
    if seed:
        print(f'ran with seed {seed}!')
    if weights:
        print(f'ran on pre-trained weights')
    return all_rewards

if __name__ == '__main__':
    # game settings
    n_episodes = 100
    learning_rate = 0.001
    rows = 7
    columns = 7
    obs_type = "pixel"  # "vector"
    max_misses = 10
    max_steps = 250
    seed = 42  # if you change this, change also above! (at very beginning)
    speed = 1.0
    boot = "MC"
    minibatch = 1
    weights = None
    # weights = 'data/weights/last_weights.h5'

    start = time.time()
    stamp = time.strftime("%d_%H%M%S", time.gmtime(start))

    learning_rates = [0.005,0.001,0.0001,0.00005]
    for learning_rate in learning_rates:
        rewards = reinforce(n_episodes,learning_rate,rows,columns,obs_type,
                  max_misses,max_steps,seed,speed,boot,weights,minibatch,stamp)
        with open("data/documentation.txt", 'a') as f:
        # export comand line output for later investigation
            f.write(f'\n\nStamp: {stamp} ... Episodes: {n_episodes}, Learning: {learning_rate}, Avg reward: {np.mean(rewards)}\n')
