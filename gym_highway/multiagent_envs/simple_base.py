import numpy as np
from gym_highway.multiagent_envs.agent import Car, Obstacle
from gym_highway.multiagent_envs.highway_world import HighwayWorld
from gym_highway.multiagent_envs.highway_core import HighwaySimulator
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        world = HighwaySimulator(**kwargs)
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 0
        # world.collaborative = True

        # initialize all agents in this world
        self._configure_world(world)
        # make initial conditions
        self.reset_world(world)
        # track rewards for info benchmark_data
        self.current_rewards = None

        return world

    def _configure_world(self, world):
        # initial positions of obstacles and agents
        policy_agents_data = [
            {'id':0, 'x':20, 'y':Constants.LANE_2_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':2},
            {'id':1, 'x':5, 'y':Constants.LANE_1_C, 'vel_x':10.0, 'vel_y':0.0, 'lane_id':1}
        ]
        scripted_agents_data = [
            {'id':2, 'x':-20, 'y':Constants.LANE_1_C, 'vel_x':13.0, 'lane_id':1, 'color':Constants.YELLOW}, 
            {'id':3, 'x':-25, 'y':Constants.LANE_2_C, 'vel_x':12.0, 'lane_id':2, 'color':Constants.YELLOW},
            {'id':4, 'x':-40, 'y':Constants.LANE_3_C, 'vel_x':10.0, 'lane_id':3, 'color':Constants.YELLOW}
        ]
        # set agent initialization data
        world.policy_agents_data = policy_agents_data
        world.scripted_agents_data = scripted_agents_data

    def reset_world(self, world):
        world.reset()

    def benchmark_data(self, world):
        """ Get ordered info for all policy agents """
        info = [None]*len(world.policy_agents_data)
        for agent in world.agents:
            data = {"rewards": self.current_rewards[agent.id],
                    "run_time": world.run_time, 
                    "num_obs_collisions": world.num_obs_collisions, 
                    "num_agent_collisions": world.num_agent_collisions}
            info[agent.id] = data
        return info
    
    def dones(self, world):
        return world.get_done()

    def rewards(self, world):
        """ Get ordered rewards for all policy agents """
        # check for collisions
        collisions = world.check_collisions()
        agent_rewards = [0.0]*len(world.policy_agents_data)
        # max reward per step is 0.0 for going at max velocity
        for agent in world.agents:
            # if in a collision assign large negative reward
            if collisions[agent.id]:
                agent_rewards[agent.id] = -250.0
            else:
                # reward of 0.0 for going max speed, negative reward otherwise
                agent_rewards[agent.id] = (agent.velocity.x / agent.max_velocity) - 1.0
        
        self.current_rewards = agent_rewards
        return agent_rewards

    def observations(self, world):
        """ 
            Get ordered observations for all policy agents

            Returns observations for all policy agents in order of agent id.
            Each observation contains relative positions (x,y) of all agents, followed by relative velocities (vx,vy).
            The first position and velocity correspond to the specific agent at that index.
            
            Returns
            -------
            [[pos_x1, pos_y1, ..., pos_xn, pos_yn, vel_x1, vel_y1, ..., vel_xn, vel_yn], ... ]
        """
        observations = [None]*len(world.policy_agents_data)
        for agent in world.agents:
            other_pos = [None]*(len(world.all_obstacles)-1) # all agents except the one in focus
            other_vel = [None]*(len(world.all_obstacles)-1)

            # get positions and velocities of all entities in this agent's reference frame
            for other_agent in world.all_obstacles:
                if other_agent is agent: continue
                
                # find place for agent info in the array
                placement_idx = other_agent.id
                if other_agent.id > agent.id: placement_idx -= 1
                
                other_pos[placement_idx] = list(other_agent.position - agent.position)
                other_vel[placement_idx] = list(other_agent.velocity - agent.velocity)
            
            ob_list = [list(agent.position) + other_pos + list(agent.velocity) + other_vel]
            obv = numpy.array(ob_list, dtype=numpy.float32).flatten()

            # ensure consistent order
            observations[agent.id] = obv
        return observations
