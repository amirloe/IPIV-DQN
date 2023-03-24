from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box

class gymEnv:
    def __init__(self, name,size):
        self.env = gym.make(name)
        self.action_space = Discrete(size)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render(mode = 'rgb_array')

    def close(self):
        self.env.close()
        
    def seed(self,num):
        self.env.seed(num)

class CartPole(gymEnv):
    def __init__(self, config=None):
        gymEnv.__init__(self,'CartPole-v1',2)


class LunarLander(gymEnv):
    def __init__(self, config=None):
        gymEnv.__init__(self,'LunarLander-v2',4)
    def set_state(self, state):
        self.env = deepcopy(state)
        return "Done"

class Breakout(gymEnv):
    def __init__(self, config=None):
        gymEnv.__init__(self,'BreakoutDeterministic-v4',4)
    def set_state(self, state):
        self.env = deepcopy(state)
        obs = self.env.render(mode='rgb_array')
        return obs