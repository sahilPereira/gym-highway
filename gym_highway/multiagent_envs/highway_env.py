import gym
import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec

from gym_highway.multiagent_envs import actions
from gym_highway.multiagent_envs.simple_base import Scenario
from gym_highway.multiagent_envs.scenario_blocking import ScenarioBlocking

# environment for all agents in the multiagent world
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world_config, num_agents=1, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_reward=False):

        scenario = Scenario()
        # scenario = ScenarioBlocking()
        # create world
        self.world = scenario.make_world(num_agents, world_config)
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(self.world.agents)
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        self.reward_callback = scenario.rewards
        self.observation_callback = scenario.observations
        self.info_callback = scenario.benchmark_data
        self.done_callback = scenario.dones
        self.shared_reward = shared_reward

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
            # action space
            self.action_space[agent.id] = spaces.Discrete(len(actions.Action))
            # observation space
            self.observation_space[agent.id] = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action_n):
        """
        Update state of the world
        """
        reward = 0.0
        num_steps = int(self.world.ticks/4)

        # set actions for scripted agents 
        # TODO: might be useful when we want the scripted agents to have a more robust script
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)

        self.agents = self.world.agents
        # set action for each agent
        for agent in self.agents:
            agent.action = actions.Action(np.argmax(action_n[agent.id]))
        
        # perform num_steps in world
        for _ in range(num_steps):
            reward_n, done_n = self._act(action_n)

            # if any agent is done, break
            if any(done_n):
                break

        # NOTE: changed from self._get_obs_norm()
        obs_n = self._get_obs_norm()
        info_n = {'n': self._get_info()}
        
        # all agents get total reward in cooperative case
        if self.shared_reward:
            reward = np.sum(reward_n)
            reward_n = [reward] * self.n
        
        # round the reward
        # TODO: removed rounding to allow for more precision
        # reward = round(reward, 3)
        return obs_n, reward_n, done_n, info_n

    def _act(self, action_n):
        """ Take one step in the world and return rewards and done signals """
        # advance world state by performing an action
        self.world.act()
        return self._get_reward(), self._get_done()

    def reset(self):
        # reset world
        self.reset_callback(self.world)

        # return observations for all agents
        # NOTE: changed from self._get_obs_norm()
        return self._get_obs_norm()

    # get info used for benchmarking
    def _get_info(self):
        if self.info_callback is None:
            return [{}]*self.n
        return self.info_callback(self.world)

    # get observation for a particular agent
    def _get_obs(self):
        if self.observation_callback is None:
            return [np.zeros(0)]*self.n
        return self.observation_callback(self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        if self.done_callback is None:
            return [False]*self.n
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self):
        if self.reward_callback is None:
            return [0.0]*self.n
        return self.reward_callback(self.world)

    def _get_obs_norm(self):
        """Get normalized observation."""
        obs_n = self._get_obs()
        for i in range(len(obs_n)):
            ob = obs_n[i]
            # clip values outside observation range
            np.clip(ob, self.low, self.high, out=ob)
            # normalize observations (min-max normalization)
            norm = (ob - self.low)/(self.high - self.low)
            obs_n[i] = norm
        return obs_n

    def close(self):
        self.world.close()
        return

    def seed(self, seed):
        self.world.seed(seed)
        return

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
