import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)

class HighwayEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.0.1"
        logging.info("HighwayEnv - Version {}".format(self.__version__))

        self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

    def __del__(self):
        pass

    def _configure_environment(self):
        pass

    def _start_viewer(self):
        pass

    def step(self, action):
        # return ob, reward, episode_over, {}
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        pass

    def reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _get_state(self):
        pass

    def seed(self, seed):
        random.seed(seed)
        np.random.seed
