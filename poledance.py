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
buffer = np.array([])
state_buffer = []
term_buffer = []
reward_buffer = []
big_R = []
counter = 0
reward = 0
done = False
budget = 1e5
ep_count = 0
ep_reward = 0



# init game
env = gym.make("CartPole-v1")  # , render_mode='human'
state, info = env.reset()
# print(observation)  # 4 state values define one state


# hyperparameters
epsilon = 0.1
output_activation = None
learning_rate = 0.1
max_buffer_length = int(1e4)
train_model_freq = 4
update_target_freq = int(100)
max_episode_length = int(1000)
batch_size = 32
gamma = 0.99

#loss_func = keras.losses.Huber()


def stable_loss(target, pred):  # implement own loss on stable target
    squared_difference = tf.square(target - pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def init_Qnet():
    model = tf.keras.Sequential([
        keras.layers.Dense(100, activation='relu',kernel_initializer='glorot_uniform',input_shape=(4,)),
        keras.layers.Dense(50, activation='relu',kernel_initializer='glorot_uniform'),
        keras.layers.Dense(nb_actions, activation=output_activation)
    ])
    return model

def make_tensor(s, list: bool):
    '''in order to be used in net.predict() method'''
    s_tensor = tf.convert_to_tensor(s)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)

def draw_action(s, net, epsilon, greedy=True):
    '''
    input: 
    - s: (4dim state) of environment
    - net: policy approximator network
    returns:
    - greedy action to act upon
    using pre-initialised -q_net-, predict an output. no training! '''
    #print(f'type of s {type(s)}')
    s_tensor = make_tensor(s, False)
    action_probs = net.predict(s_tensor, verbose=0)

    if greedy or np.random.uniform(0.,1) > epsilon:
        return np.argmax(action_probs)
    else:
        return np.random.randint(0,nb_actions)

def net_update(net, target_net, states, rewards, gamma):
    states = make_tensor(states, list=True)
    target_output = target_net.predict(states, verbose=0)
    target_val = rewards + gamma *  np.max(target_output, axis=1)  # target value to compare to
    
    history = net.fit(states,target_val,batch_size=batch_size, verbose=0)
    loss = history.history['loss'][0]
    #print('loss:', loss)

    '''
    with tf.GradientTape() as tape:
        q_values = net(states, training=True)
        loss = stable_loss(q_values[action],target_val[-1])

    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    '''
    return net, loss


q_net = init_Qnet()
q_net.summary()
target_q_net = q_net  # clone into target network
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
q_net.compile(optimizer,loss=stable_loss)

#q_net.summary()
#q_net.predict(observation)

while True:
    '''work in progress..'''

    #print(f"episode {ep_count} reward {ep_reward}")
    ep_count += 1
    ep_reward = 0
    if ep_count % 10 == 0:
        print(f"mean reward of last 10 {np.mean(big_R[-10:])}")

    for timestep in range(1,max_episode_length):
        # print(f"time {timestep} ({ep_count})")

        # remove random step when buffer too long
        if len(state_buffer) > max_buffer_length:
            d = np.random.randint(len(state_buffer))
            term_buffer.pop(d)
            reward_buffer.pop(d)
            state_buffer.pop(d)

        # draw action
        action = draw_action(state, q_net, epsilon=epsilon, greedy=False)
        next_state, r, term, trunk, info = env.step(action=action)
        #print(f'reward {r} at time {timestep} ({ep_count}), trunk {trunk}, term {term}')

        ep_reward += r
        state_buffer.append(state)
        term_buffer.append([term,trunk])
        reward_buffer.append(r)
        state = next_state

        # sample buffer
        if len(state_buffer) <= batch_size:
            states = np.array(state_buffer)
            rewards = reward_buffer
        else:
            choice = np.random.choice(np.arange(len(state_buffer)), size = (batch_size,), replace=False)
            states = np.array(state_buffer)[choice]
            rewards = np.array(reward_buffer)[choice]

        if timestep % train_model_freq == 0:
            q_net, loss = net_update(q_net, target_q_net, states=states, rewards=rewards, gamma=gamma)
        if timestep % update_target_freq == 0:
            target_q_net = q_net
            print('target update!')
        
        if timestep % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (timestep, float(loss))
            )

        if term:
            #print('term')
            #print(f'episode reward:{ep_reward}')
            big_R += [ep_reward]
            state, info = env.reset()
            break

        elif trunk:
            #print('trunk')
            #print(f'episode reward:{ep_reward}')
            state, info = env.reset()
            break

    if ep_reward > 100 or ep_count > 100:
        print(f'much reward! ep_reward = {ep_reward}')
        break