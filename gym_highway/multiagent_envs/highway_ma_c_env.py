import logging

import gym
import numpy as np
from gym import error, spaces

from gym_highway.multiagent_envs.highway_env import MultiAgentEnv
from gym_highway.multiagent_envs.simple_base import Scenario

logger = logging.getLogger(__name__)

class MultiAgentEnvContinuous(MultiAgentEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world_config, num_agents=1, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_reward=False):

        super(MultiAgentEnvContinuous, self).__init__(world_config, num_agents, reset_callback, 
                reward_callback, observation_callback, info_callback, done_callback, shared_reward)

        # update world for continuous action space
        self.world.continuous_ctrl = True

        num_entities = len(self.world.entities)
        # observations contain (accel, steering), all positions (x,y), followed by all velocities (vx,vy)
        # using relative positions and velocities for bounding
        control_low = np.array([-5.0, -30.0])
        control_high = np.array([5.0, 30.0])

        # first bound is for raw min position
        pos_low = np.array([0.0, 0.0]+[-60.0, -8.0]*(num_entities-1)).flatten()
        vel_low = np.array([-20.0, -20.0]*num_entities).flatten()
        # first bound is for raw max position
        pos_high = np.array([1200.0, 8.0]+[60.0, 8.0]*(num_entities-1)).flatten()
        vel_high = np.array([20.0, 20.0]*num_entities).flatten()
        
        self.low = np.concatenate((control_low, pos_low, vel_low))
        self.high = np.concatenate((control_high, pos_high, vel_high))

        assert len(self.low) == len(control_low)+len(pos_low)+len(vel_low)
        assert len(self.high) == len(control_high)+len(pos_high)+len(vel_high)

        # configure spaces
        self.action_space = [None]*self.n
        self.observation_space = [None]*self.n
        obs_dim = len(self.observation_callback(self.world)[0])
        for agent in self.agents:
            # action space continuous
            self.action_space[agent.id] = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            # observation space
            self.observation_space[agent.id] = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action_n):
        """
        Update state of the world
        """
        num_steps = int(self.world.ticks/4)
        self.agents = self.world.agents
        # set action for each agent
        for agent in self.agents:
            agent.action = action_n[agent.id]
        
        # perform num_steps in world
        for _ in range(num_steps):
            reward_n, done_n = self._act(action_n)

            # if any agent is done, break
            if any(done_n):
                break

        # NOTE: changed from self._get_obs_norm()
        obs_n = self._get_obs()
        info_n = {'n': self._get_info()}
        
        # all agents get total reward in cooperative case
        if self.shared_reward:
            reward = np.sum(reward_n)
            reward_n = [reward] * self.n
        
        return obs_n, reward_n, done_n, info_n

