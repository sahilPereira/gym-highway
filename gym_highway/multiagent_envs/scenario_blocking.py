import numpy as np

import models.config as Config
from gym_highway.multiagent_envs import highway_constants as Constants
from gym_highway.multiagent_envs.agent import Car, Obstacle
from gym_highway.multiagent_envs.highway_core import HighwaySimulator
# from gym_highway.multiagent_envs.highway_world import HighwayWorld
from gym_highway.multiagent_envs.multiagent.scenario import BaseScenario


class ScenarioBlocking(BaseScenario):
    def make_world(self, num_agents, world_kw_args):
        world = HighwaySimulator(**world_kw_args)
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents
        num_obstacles = 3
        # world.collaborative = True

        # initialize all agents in this world
        self.configure_world(world, num_agents, num_obstacles)
        # make initial conditions
        self.reset_world(world)
        # track rewards for info benchmark_data
        self.current_rewards = None

        return world

    def configure_world(self, world, n_agents, n_obstacles):
        # initial positions of obstacles and agents
        # initialize in x formation
        policy_agents_data = [
            {'id':0, 'x':10, 'y':Constants.LANE_1_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':1},
            {'id':1, 'x':10, 'y':Constants.LANE_3_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':3},
            {'id':2, 'x':7.5, 'y':Constants.LANE_2_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':2},
            {'id':3, 'x':5, 'y':Constants.LANE_1_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':1},
            {'id':4, 'x':5, 'y':Constants.LANE_3_C, 'vel_x':0.0, 'vel_y':0.0, 'lane_id':3}
        ]
        scripted_agents_data = [
            {'id':n_agents, 'x':-20, 'y':Constants.LANE_1_C, 'vel_x':7.0, 'lane_id':1, 'color':Constants.YELLOW}, 
            {'id':n_agents+1, 'x':-25, 'y':Constants.LANE_2_C, 'vel_x':5.0, 'lane_id':2, 'color':Constants.YELLOW},
            {'id':n_agents+2, 'x':-40, 'y':Constants.LANE_3_C, 'vel_x':6.0, 'lane_id':3, 'color':Constants.YELLOW}
        ]
        # set agent initialization data
        world.policy_agents_data = policy_agents_data[:n_agents]
        world.scripted_agents_data = scripted_agents_data[:n_obstacles]

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

                # penalize follower
                if agent.position.x < 10.0:
                    agent_rewards[agent.id] -= 0.5
        
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
            # all agents except the one in focus and max 3 obstacles
            # 3 obs - 1 ego agent = 2 extra
            other_pos = [None]*(len(world.agents)+2)
            other_vel = [None]*(len(world.agents)+2)

            lane_obs_near = [None for _ in range(Constants.NUM_LANES)]
            # get closest obstacle

            # TEST: make sure we have sufficient obstacles
            assert len(world.scripted_agents) >= 3
            for obj in world.scripted_agents:
                if obj is agent: continue
                
                lane_idx = obj.lane_id-1
                if lane_obs_near[lane_idx] is not None:
                    # compute absolute difference in position between ref and obs
                    pos_diff = abs(agent.raw_position.x - lane_obs_near[lane_idx].raw_position.x)
                    new_pos_diff = abs(agent.raw_position.x - obj.raw_position.x)
                    # if current obs is closer to ref vehicle
                    if new_pos_diff < pos_diff:
                        lane_obs_near[lane_idx] = obj
                else:
                    # if no object in this lane
                    lane_obs_near[lane_idx] = obj

            # store closest obstacles
            for lane in range(Constants.NUM_LANES):
                obj = lane_obs_near[lane]
                if obj is agent:
                    raise Exception("Obstacle object should not equal reference agent")
                if obj is None:
                    raise Exception("Obj is None, lane_obs_near: {}".format(lane_obs_near))
                # find place for agent info in the array
                placement_idx = obj.id
                if obj.id > agent.id: placement_idx -= 1
                
                other_pos[placement_idx] = list(obj.raw_position - agent.raw_position)
                other_vel[placement_idx] = list(obj.velocity - agent.velocity)

            # get positions and velocities of all other agents in this agent's reference frame
            for other_agent in world.agents:
                if other_agent is agent: continue
                
                # find place for agent info in the array
                placement_idx = other_agent.id
                if other_agent.id > agent.id: placement_idx -= 1
                
                other_pos[placement_idx] = list(other_agent.raw_position - agent.raw_position)
                other_vel[placement_idx] = list(other_agent.velocity - agent.velocity)
            
            ob_list = [(agent.acceleration, agent.steering)] + [agent.raw_position] + other_pos + [agent.velocity] + other_vel
            
            # ob_list should contain accel/steering/pos/vel of current agent and pos/vel of all other agents
            assert len(ob_list) == len(other_pos)+len(other_vel)+3
            obv = np.array(ob_list, dtype=np.float32).flatten()

            # ensure consistent order
            observations[agent.id] = obv
        return observations
