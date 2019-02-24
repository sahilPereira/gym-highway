import numpy as np
from gym_highway.multiagent_envs.agent import Car, Obstacle
from gym_highway.multiagent_envs.highway_world import HighwayWorld
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        world = HighwayWorld(**kwargs)
        return world

    def reset_world(self, world):
        pass

    def benchmark_data(self, agent, world):
        pass

    def reward(self, agent, world):
        pass

    def observation(self, agent, world):
        pass
