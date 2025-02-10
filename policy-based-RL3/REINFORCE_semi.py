import numpy as np
from catch import Catch
import tensorflow as tf
from collections import deque
from Helper import time_it, print_it, save_params, write_to_doc
import time
from keras.utils.vis_utils import plot_model

    
'''
    TO DO: 
        ask: 
            - how many epochs/steps should we aim for? -> 200k-500k steps

        implement:
            - checkpoints (tensorboard)

        issue:

        Experiments:
        1) REINFORCE, actor critic bootstrap, ac baseline, ac baseline + bootstrap (aren't the last two the same?)
            put them all in one plot, average over 5 runs, 1000 episodes
        2) vary size (5x5 converges faster): compare 5x5 vs 7x7, averaged over 5 runs and 1000-2000 traces
        4) Other environment variations.
    '''


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']


class Actor():
    def __init__(self, learning_rate: float = 0.01, arch: int = 1, observation_type: str = "pixel",
                 rows=7, columns=7, boot: str = "MC", n_step: int = 1, saved_weights: str = None,
                 seed=None, critic: bool = False, eta: float = 0.01, baseline: bool = False, training: bool = True, ppo: bool = False):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.boot = boot
        self.n_step = n_step
        self.critic = critic  # if true, it is a critic network
        self.baseline = baseline
        self.eta = eta
        self.training = training
        self.gamma = 0.99
        self.training = training
        
        self.ppo = ppo
        self.clip_pram = 0.8

        # network parameters
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        init2 = tf.keras.initializers.GlorotNormal(seed=self.seed)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if (observation_type == 'pixel'):
            input_shape = (columns, rows, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)
        if arch == 1:
            input = tf.keras.layers.Input(shape=input_shape)
            flatten = tf.keras.layers.Flatten()(input)
            dense = tf.keras.layers.Dense(
                64, activation=activ_func, kernel_initializer=init)(flatten)
            batchNorm = tf.keras.layers.BatchNormalization()(dense)
            dense2 = tf.keras.layers.Dense(
                32, activation=activ_func, kernel_initializer=init2)(batchNorm)
            # batchNorm2 = tf.keras.layers.BatchNormalization()(dense2)
            dropout = tf.keras.layers.Dropout(0.2)(dense2)
            dense3 = tf.keras.layers.Flatten()(dropout)
        if critic:
            output_value = tf.keras.layers.Dense(
                1, activation='linear')(dense3)
            self.model = tf.keras.models.Model(
                inputs=input, outputs=[output_value])
        else:
            output_actions = tf.keras.layers.Dense(
                3, activation='softmax')(dense3)
            self.model = tf.keras.models.Model(
                inputs=input, outputs=[output_actions])

        self.model.summary()
        if saved_weights:
            print('## Working with pre-trained weights ##')
            self.model.load_weights(saved_weights)
        if not training:
            print('## Not training ##')
        # plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def bootstrap(self, t, rewards, values=None):
        if self.boot == "MC":
            rewards = rewards[t:]
            return self.gamma**np.arange(0, len(rewards)) @ rewards
        elif self.boot == "n_step":
            lim = min(t+self.n_step, len(rewards))
            rewards = rewards[t:lim]
            return self.gamma**np.arange(0, len(rewards)) @ rewards + self.gamma**self.n_step * values[lim-1]

    def gain_fn_entropy(self, prob_out=None, Q=None, actions=None):
        # rewrites [-1,0,1] into [0,1,2]
        actions = np.where(actions == -1, 0, np.where(actions == 0, 1, 2))
        mask = tf.one_hot(actions, 3)
        prob_out = tf.reduce_max(mask * prob_out, axis=1)

        gain = tf.tensordot(tf.constant(-1 * np.ones(len(Q)) * Q, dtype=tf.float32),
                            tf.math.log(prob_out), 1)
        # -sum_i ( pi * log(pi) )
        gain -= self.eta * tf.tensordot(prob_out, tf.math.log(prob_out), 1)
        return gain

    def update_weights(self, memories):
        '''got code structure from https://keras.io/guides/writing_a_training_loop_from_scratch/'''
        if not self.training:
            return

        gradients = []
        for k, memory in enumerate(memories):
            # memory['number_of_minibatches', 'values']
            states, actions, rewards, values = [np.array([experience[field_index]
                                                          for experience in memory])
                                                for field_index in range(4)]
            if self.critic:
                ## I just wanna see where it is going
                left = np.sum(np.where(actions == -1, 1, 0))
                idle = np.sum(np.where(actions == 0, 1, 0))
                right = np.sum(np.where(actions == 1, 1, 0))
                print(f"left {left}, idle {idle}, right {right}")

            QA_values = [self.bootstrap(t, rewards, values)
                         for t in range(len(rewards))]
            if self.baseline:
                QA_values = [QA_values[i]-values[i]
                             for i in range(len(rewards))]

            states = tf.convert_to_tensor(states)
            with tf.GradientTape() as tape:
                if self.critic:
                    values = self.model(states)
                    gain_value = tf.losses.mean_squared_error(QA_values, values)
                else:
                    probs_out = self.model(states)
                    gain_value = self.gain_fn_entropy(
                        prob_out=probs_out, Q=QA_values, actions=actions)
                    
                    if self.ppo:
                        old_probs = tf.reshape(old_probs, (len(old_probs), 3))
                        actions = np.where(actions == -1, 0, np.where(actions == 0, 1, 2))
                        mask = tf.one_hot(actions, 3)
                        probs_out = tf.reduce_max(mask * probs_out, axis=1)
                        old_probs = tf.reduce_max(mask * old_probs, axis=1)
                        
                        c_loss = tf.losses.mean_squared_error(QA_values, values)
                        ratio = tf.math.divide(probs_out, old_probs)
                        
                        QA_values = tf.convert_to_tensor(QA_values, dtype=tf.float32)
                        
                        s1 = tf.math.multiply(ratio, QA_values)
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), QA_values)
                        
                        s12 = tf.reduce_mean(tf.math.minimum(s1, s2))
                        gain_value = tf.math.negative(s12 + 0.5*c_loss - gain_value)
   

            gradients.append(tape.gradient(gain_value,self.model.trainable_weights))
        
        # average of grads
        average_grads = []
        for grads in zip(*gradients):
            avg = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)
            average_grads.append(avg)

        self.optimizer.apply_gradients(
            zip(average_grads, self.model.trainable_weights))
        # norm_grads = [tf.norm(grad) for grad in grads]
        return average_grads

    def reshape_state(self, state):
        if self.observation_type == "pixel":
            return state.reshape(1, self.columns, self.rows, 2)
        elif self.observation_type == "vector":
            return state.reshape(1, 3)

    def unshape_state(self, state):
        '''same is done by squeeze'''
        if self.observation_type == "pixel":
            return state.reshape(self.columns, self.rows, 2)
        elif self.observation_type == "vector":
            return state.reshape(3)


@save_params
@time_it
def reinforce(n_episodes: int = 50, learning_rate: float = 0.001, rows: int = 7, columns: int = 7,
              obs_type: str = "pixel", max_misses: int = 10, max_steps: int = 250, seed: int = None,
              n_step: int = 5, speed: float = 1.0, boot: str = "MC", P_weights: str = None, V_weights: str = None,
              minibatch: int = 1, eta: float = 0.01, stamp: str = None, baseline: bool = False, training: bool = True, ppo: bool = False):
    text = f"\n\nRunning on {reinforce.params}\n"
    print(text)
    write_to_doc(text=text)
    if boot == "MC":
        n_step = max_steps

    rng = np.random.default_rng(seed=seed)

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=obs_type, seed=None)

    all_rewards = []
    actor = Actor(learning_rate, boot=boot, n_step=n_step, rows=rows, columns=columns,
                  observation_type=obs_type, saved_weights=P_weights,
                  seed=seed, eta=eta, baseline=baseline, training=training, ppo=ppo)

    if boot == 'n_step' or baseline:
        critic = Actor(0.05, boot=boot, n_step=n_step, rows=rows, columns=columns,
                       observation_type=obs_type, saved_weights=V_weights, seed=seed,
                       critic=True, eta=eta, training=training, baseline=False, ppo=ppo)
    count = 0
    memory = [deque(maxlen=max_steps) for _ in range(minibatch)]
    ep = 0
    all_grads = []
    while ep < n_episodes:
        # for ep in range(n_episodes):  # n_episodes = 500
        if len(all_rewards) > 0 and np.mean(all_rewards[-20:]) > 15:
            # turn off selection of best-performing when we already reached high rewards
            minibatch = 2
            memory = [deque(maxlen=max_steps) for _ in range(minibatch)]

        for mem in memory:
            mem.clear()
        ep += min(minibatch,2)
        for m in range(minibatch):
            state = actor.reshape_state(env.reset())
            # generate full trace
            for T in range(max_steps):
                action_p = actor.model.predict(state, verbose=0)

                if actor.boot == "n_step" or baseline:
                    value = critic.model.predict(state, verbose=0)
                    value = tf.squeeze(value)
                elif actor.boot == "MC":
                    value = None

                action = rng.choice(
                    ACTION_EFFECTS, p=action_p.reshape(3,))
                next_state, r, done = env.step(action)
                count += 1
                next_state = actor.reshape_state(next_state)
                # env.render(0.2)

                # take out the extra "1" dimensions
                memory[m].append((tf.squeeze(state), action, r, value))

                if done:
                    break
                state = next_state
            # trace is finished
            # print(f'{ep}, step {count}, reward: {ep_reward}')

            # memory['number_of_minibatches', 'values']
        
        avg_total_rewards = []
        total_rewards = []
        for mem in memory:
            rewards = np.array([experience[2] for experience in mem])
            avg_total_rewards.append(np.mean(rewards))
            total_rewards.append(rewards)

        best_memory = []
        ## Choose the best two average total rewards from memory buffer
        for _ in range(min(minibatch,2)):  # this decides over how many we are going to average
            rewards_max_index = np.argmax(avg_total_rewards)  # get index of best performing
            avg_total_rewards[rewards_max_index] = -9.  # make sure it's not chosen again
            best_memory.append(memory[rewards_max_index])  # save this memory to use in update
            all_rewards.append(np.sum(total_rewards[rewards_max_index]))  # export rewards for plot later

        print(f"{ep}, step {count}, rewards: {all_rewards[-2:]}")
        if actor.boot == 'n_step' or baseline:
            critic.update_weights(best_memory)
        grads = actor.update_weights(best_memory)
        # print("GRADS", grads)

        ep_grad_avg = [tf.math.reduce_mean(grad) for grad in grads]
        all_grads.append(np.mean(ep_grad_avg))

        if ep % 10 == 0 and ep > 0:
            np.save(f'data/rewards/tmp_reward', all_rewards)
            np.save(f'data/grads/tmp_grads', np.array(all_grads))
        if ep % 50 == 0 and ep >= 100:
            actor.model.save_weights(f'data/weights/w_P_{stamp}.h5')
            np.save(f'data/grads/g_{stamp}', np.array(all_grads))
            np.save(f'data/rewards/r_{stamp}', all_rewards)
            if boot == "n_step" or baseline:
                critic.model.save_weights(f'data/weights/w_V_{stamp}.h5')
        print()

    actor.model.save_weights(f'data/weights/w_P_{stamp}.h5')
    if boot == "n_step" or baseline:
        critic.model.save_weights(f'data/weights/w_V_{stamp}.h5')
    np.save(f'data/rewards/r_{stamp}', all_rewards)
    write_to_doc(f'{stamp} ... Avg reward: {np.mean(all_rewards)} \n')
    np.save(f'data/grads/g_{stamp}', np.array(all_grads))
    return all_rewards


if __name__ == '__main__':
    # game settings
    n_episodes = 150
    learning_rate = 0.01
    rows = 7
    columns = 7
    obs_type = "pixel"  # "vector" or "pixel"
    max_misses = 10
    max_steps = 250
    n_step = 5
    speed = 1.
    boot = "n_step"  # "n_step" or "MC"
    minibatch = 4  # but only 2 best ones are used 
    P_weights = None
    V_weights = None
    baseline = True
    eta = 0.001
    P_weights = 'data/weights/w_P_08_085540.h5'
    V_weights = 'data/weights/w_V_08_085540.h5'
    # use '27_230853','28_002357' next
    training = True
    ppo = False

    for _ in range(1):
        seed = np.random.randint(100)  # 25 went well
        stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
        rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                            max_misses, max_steps, seed, n_step, speed, boot,
                            P_weights, V_weights, minibatch, eta, stamp, baseline, training)
