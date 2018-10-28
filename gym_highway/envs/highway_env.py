import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import random
import numpy as np

# import multi_lane_sim as mls
from gym_highway.envs.multi_lane_sim import HighwaySimulator, Action, Constants
import logging
logger = logging.getLogger(__name__)

ACTION_LOOKUP = {
    0 : Action.LEFT,
    1 : Action.RIGHT,
    2 : Action.MAINTAIN,
    3 : Action.ACCELERATE,
    4 : Action.DECELERATE,
}

class HighwayEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, manual=False, inf_obs=True, save=False, render=False):
        self.__version__ = "0.0.1"
        logging.info("HighwayEnv - Version {}".format(self.__version__))

        self.env = self._configure_environment(manual, inf_obs, save, render)

        self.action_space = spaces.Discrete(len(Action))

        num_vehicles = len(self.env.cars_list) + 3 # max 3 obstacles on road at any given time

        low = np.array([-100,0,0,-20]*num_vehicles).flatten()
        high = np.array([200,8,20,20]*num_vehicles).flatten()
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # self.action_episode_memory = []

    def _configure_environment(self, manual, inf_obs, save, render):
        # initial positions of obstacles and agents
        obstacle_1 = {'id':100, 'x':-20, 'y':Constants.LANE_1_C, 'vel_x':13.0, 'lane_id':1, 'color':Constants.YELLOW}
        obstacle_2 = {'id':101, 'x':-25, 'y':Constants.LANE_2_C, 'vel_x':12.0, 'lane_id':2, 'color':Constants.YELLOW}
        obstacle_3 = {'id':102, 'x':-40, 'y':Constants.LANE_3_C, 'vel_x':10.0, 'lane_id':3, 'color':Constants.YELLOW}
        obstacle_list = [obstacle_1, obstacle_2, obstacle_3]

        car_1 = {'id':0, 'x':20, 'y':Constants.LANE_2_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':2}
        # car_2 = {'id':1, 'x':5, 'y':LANE_1_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':1}
        # car_3 = {'id':2, 'x':5, 'y':LANE_2_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':2}
        # car_4 = {'id':3, 'x':20, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        # car_5 = {'id':4, 'x':5, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        cars_list = [car_1]

        highwaySim = HighwaySimulator(cars_list, obstacle_list, manual, inf_obs, save, render)
        return highwaySim

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        reward = 0.0

        num_steps = 15 #int(60*0.25)
        for _ in range(num_steps):
            reward += self._take_action(action)
        ob = self._get_state()

        return ob, reward, self.env.is_episode_over(), self.env.get_info()

    def _take_action(self, action):
        """ Converts the action space into an Enum action. """
        action_type = ACTION_LOOKUP[action]
        return self.env.act(action_type)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.env.reset()
        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        ob = self.env.get_state()
        
        # clip values outside observation range
        np.clip(ob, self.observation_space.low, self.observation_space.high, out=ob)

        # normalize observations (min-max normalization)
        norm = (ob - self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
        return norm

    def close(self):
        self.env.close()
        return

    def seed(self, seed):
        self.env.seed(seed)
        # random.seed(seed)
        # np.random.seed
