import logging

import numpy as np

from gym import error, spaces
from gym_highway.envs.highway_env import HighwayEnv

logger = logging.getLogger(__name__)

class HighwayEnvContinuous(HighwayEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, manual=False, inf_obs=True, save=False, render=True, real_time=False):
        HighwayEnv.__init__(self, manual=manual, inf_obs=inf_obs, save=save, render=render, real_time=real_time)
        logging.info("HighwayEnvContinuous - Version {}".format(self.__version__))

        # update env for continuous action space
        self.env.continuous_ctrl = True
        # define action space
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)
        # define observation space
        num_vehicles = len(self.env.cars_list) + 3 # max 3 obstacles on road at any given time
        low = np.array([-60.0, 0.0, 0.0, -20.0]*num_vehicles).flatten()
        high = np.array([60.0, 8.0, 20.0, 20.0]*num_vehicles).flatten()
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        reward = 0.0
        # allow for actions at 4Hz
        num_steps = int(self.env.ticks/4)
        for _ in range(num_steps):
            reward = self.env.act(action)
            # if action leads to crash, end the run
            if reward < -1.0:
                break
        ob = self._get_state()

        # round the reward
        reward = round(reward, 3)
        return ob, reward, self.env.is_episode_over(), self.env.get_info()
