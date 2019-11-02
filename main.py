import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

env_name ="CartPole-v0"
env = gym.make(env_name)

# print(env.observation_space)
# print(env.action_space)
class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth= action_size)
        self.hidden = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate =0.001).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        
        session.run(self.optimizer, feed_dict = feed)
    
    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict = {self.state_in: state})
        return q_state


class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.97

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action = np.argmax(q_state)
        return action

    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.get_q_state(self.sess, [next_state])
        q_next_state = (1-done) * q_next_state
        q_target = reward + self.gamma *np.max(q_next_state)
        self.q_network.update_model(self.sess, [state], [action], [q_target])

    def __del__(self):
        self.sess.close()







agent = DQNAgent(env)
episodes = 200
for ep in range(episodes):
    state= env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state
    print(f"Episodes: {ep}, total_reward: {round(total_reward)}")
    