import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import random
import numpy as np

import multi_lane_sim as mls
import logging
logger = logging.getLogger(__name__)

class HighwayEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, manual=False, inf_obs=True, save=True):
        self.__version__ = "0.0.1"
        logging.info("HighwayEnv - Version {}".format(self.__version__))

        self.env = self._configure_environment(manual, inf_obs, save)

        self.action_space = spaces.Discrete(len(mls.Action))

        num_vehicles = len(self.env.cars_list) + 3 # max 3 obstacles on road at any given time
        # self.observation_space = spaces.Box(low=-100, high=100, shape=(num_vehicles, 4))
        self.observation_space = spaces.Box(np.array([-100,0,0,0]), np.array([100,8,25,25]))

        # Store what the agent tried
        # self.curr_episode = -1
        # self.action_episode_memory = []

    def __del__(self):
        pass

    def _configure_environment(self, manual, inf_obs, save):
        # initial positions of obstacles and agents
        obstacle_1 = {'id':100, 'x':-20, 'y':LANE_1_C, 'vel_x':13.0, 'lane_id':1, 'color':YELLOW}
        obstacle_2 = {'id':101, 'x':-25, 'y':LANE_2_C, 'vel_x':12.0, 'lane_id':2, 'color':YELLOW}
        obstacle_3 = {'id':102, 'x':-40, 'y':LANE_3_C, 'vel_x':10.0, 'lane_id':3, 'color':YELLOW}
        obstacle_list = [obstacle_1, obstacle_2, obstacle_3]

        car_1 = {'id':0, 'x':20, 'y':LANE_2_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':2}
        car_2 = {'id':1, 'x':5, 'y':LANE_1_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':1}
        car_3 = {'id':2, 'x':5, 'y':LANE_2_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':2}
        car_4 = {'id':3, 'x':20, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        car_5 = {'id':4, 'x':5, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        cars_list = [car_1, car_2, car_3, car_4, car_5]

        highwaySim = mls.HighwaySimulator(cars_list, obstacle_list, manual, inf_obs, save)
        return highwaySim

    def _start_viewer(self):
        pass

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

        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_banana_sold, {}


        # atari game
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        self.price = ((float(self.MAX_PRICE) /
                      (self.action_space.n - 1)) * action)

        chance_to_take = get_chance(self.price)
        banana_is_sold = (random.random() < chance_to_take)

        if banana_is_sold:
            self.is_banana_sold = True

        remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = (remaining_steps <= 0)
        throw_away = time_is_over and not self.is_banana_sold
        if throw_away:
            self.is_banana_sold = True  # abuse this a bit
            self.price = 0.0
    
    def _get_reward(self):
        """Reward is given for a sold banana."""
        if self.is_banana_sold:
            return self.price - 1
        else:
            return 0.0

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_banana_sold = False
        self.price = 1.00
        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed
