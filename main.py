import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

env_name ="CartPole-v0"
env = gym.make(env_name)

# print(env.observation_space)
# print(env.action_space)
env.reset()
for ep in range(200):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    