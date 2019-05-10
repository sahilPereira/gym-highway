import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import random
import numpy as np

from gym_highway.multiagent_envs import actions
from gym_highway.multiagent_envs.highway_core import HighwaySimulator
# import gym_highway.multiagent_envs.highway_constants as Constants
from gym_highway.multiagent_envs import highway_constants as Constants

import logging
logger = logging.getLogger(__name__)

class HighwayWorld(object):
    # metadata = {'render.modes': ['human']}

    def __init__(self, manual=False, inf_obs=True, save=False, render=True, real_time=False):
        # self.__version__ = "0.0.1"
        # logging.info("HighwayWorld - Version {}".format(self.__version__))

        self.env = self._configure_environment(manual, inf_obs, save, render, real_time)
        self.action_space = spaces.Discrete(len(actions.Action))
        num_vehicles = len(self.env.cars_list) + 3 # max 3 obstacles on road at any given time

        low = np.array([-60.0, 0.0, 0.0, -20.0]*num_vehicles).flatten()
        high = np.array([60.0, 8.0, 20.0, 20.0]*num_vehicles).flatten()
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def _configure_environment(self, manual, inf_obs, save, render, real_time):
        # initial positions of obstacles and agents
        obstacle_1 = {'id':100, 'x':-20, 'y':Constants.LANE_1_C, 'vel_x':13.0, 'lane_id':1, 'color':Constants.YELLOW}
        obstacle_2 = {'id':101, 'x':-25, 'y':Constants.LANE_2_C, 'vel_x':12.0, 'lane_id':2, 'color':Constants.YELLOW}
        obstacle_3 = {'id':102, 'x':-40, 'y':Constants.LANE_3_C, 'vel_x':10.0, 'lane_id':3, 'color':Constants.YELLOW}
        obstacle_list = [obstacle_1, obstacle_2, obstacle_3]

        car_1 = {'id':0, 'x':20, 'y':Constants.LANE_2_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':2}
        # car_2 = {'id':1, 'x':5, 'y':LANE_1_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':1}
        # car_3 = {'id':2, 'x':5, 'y':LANE_2_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':2}
        # car_4 = {'id':3, 'x':20, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        # car_5 = {'id':4, 'x':5, 'y':LANE_3_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':3}
        cars_list = [car_1]

        highwaySim = HighwaySimulator(cars_list, obstacle_list, manual, inf_obs, save, render, real_time)
        return highwaySim

    # TODO: not sure if this communication channel is required
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise   

    def step(self):
        """
        Update state of the world
        """
        reward = 0.0
        num_steps = int(self.env.ticks/4)

        # set actions for scripted agents 
        # TODO: might be useful when we want the scripted agents to have a more robust script
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)

        for _ in range(num_steps):
            reward = self._take_action()

            # if action leads to crash, end the run
            if reward < -1.0:
                break

        ob = self._get_state()
        # round the reward
        reward = round(reward, 3)
        return ob, reward, self.env.is_episode_over(), self.env.get_info()

    def _take_action(self):
        """ All agents execute their selected actions """
        return self.env.act()

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
