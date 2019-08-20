import math
import os
import random
from math import ceil, copysign, degrees, radians, tan
from random import randrange

import numpy
import pygame
from pygame.math import Vector2

from gym_highway.multiagent_envs import actions
from gym_highway.multiagent_envs import highway_constants as Constants
from gym_highway.multiagent_envs.agent import Car, Obstacle

class HighwaySimulator:
    def __init__(self, manual=False, inf_obs=False, save=False, render=False, real_time=False, continuous=False):
        pygame.init()
        width = Constants.WIDTH
        height = Constants.HEIGHT
        if render:
            pygame.display.set_caption("Car tutorial")
            self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60.0 if real_time else 36.0
        self.framerate = self.ticks*2 if real_time else 0.0
        self.exit = False

        # simulation options
        self.policy_agents_data = None
        self.scripted_agents_data = None
        self.is_manual = manual
        self.inf_obstacles = inf_obs
        self.is_data_saved = save
        self.render = render
        self.run_duration = 60.0 # 60 seconds
        self.continuous_ctrl = continuous

        # list of agents and entities (can change at execution-time!)
        self.agents = None
        self.scripted_agents = None
        self.all_obstacles = None
        # communication channel dimensionality
        self.dim_c = 0

        self.reference_car = None
        self.action_timer = 0.0
        self.log_timer = 0.0
        self.continuous_time = 0.0
        self.num_obs_collisions = 0
        self.num_agent_collisions = 0
        self.collision_count_lock = True
        self.run_time = 0.0

        self.is_done = None
        self.reward = None
        self.lateral_offset = False
        
        # reset simulator states
        # self.reset()

    # return all entities in the world
    @property
    def entities(self):
        return self.all_obstacles
    
    def set_agents(self, agents):
        """ Set all policy agents used in this world """
        for agent in agents:
            self.agents.add(agent)
            self.all_obstacles.add(agent)

    def set_scripted_agents(self, scripted_agents):
        """ Set all scripted agents used in this world """
        for agent in scripted_agents:
            self.scripted_agents.add(agent)
            self.all_obstacles.add(agent)

    def loadBackground(self):
        bkgd = pygame.image.load('gym_highway/envs/roadImg.png').convert()
        bkgd = pygame.transform.scale(bkgd, (Constants.WIDTH, Constants.HEIGHT))
        return bkgd

    def displayScore(self, steering, car_angle):
        font = pygame.font.SysFont(None, 25)
        text = font.render("Collision [obstacle]: "+str(steering), True, Constants.WHITE)
        text_angle = font.render("Collision [agent]: "+str(car_angle), True, Constants.WHITE)
        self.screen.blit(text, (0,0))
        self.screen.blit(text_angle, (0,25))

    def displayVel(self, velocity):
        font = pygame.font.SysFont(None, 25)
        text = font.render("Velocity: "+str(velocity), True, Constants.WHITE)
        self.screen.blit(text, (0,50))

    def displayPos(self, position):
        font = pygame.font.SysFont(None, 25)
        text = font.render("X: "+str(position.x)+", Y: "+str(position.y), True, Constants.WHITE)
        self.screen.blit(text, (0,100))

    def displayAction(self, action):
        font = pygame.font.SysFont(None, 25)
        text = font.render("Action: "+str(action), True, Constants.WHITE)
        self.screen.blit(text, (0,75))

    def updateSprites(self, vehicles):
        for auto in vehicles:
            rotated = pygame.transform.rotate(auto.image, auto.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, auto.position * Constants.ppu - (rect.width / 2, rect.height / 2))

    def executeAction(self, agent, all_obstacles, dt):
        selected_action = agent.action
        if self.continuous_ctrl:
            self.executeActionContinuous(selected_action, agent, all_obstacles, dt)
        else:
            self.executeActionDiscrete(selected_action, agent, all_obstacles)

    # execute the given action for the specified leader
    def executeActionDiscrete(self, selected_action, leader, all_obstacles):
        if (selected_action == Action.ACCELERATE) and not leader.do_accelerate:
            self.accelerate(leader)
        elif (selected_action == Action.MAINTAIN) and not leader.do_maintain:
            self.maintain(leader, all_obstacles)
        # elif (selected_action == Action.DECELERATE) and not leader.do_decelerate:
        #     self.decelerate(leader)

        leader.acceleration = max(-leader.max_acceleration, min(leader.acceleration, leader.max_acceleration))

        if (selected_action == Action.RIGHT) and not leader.right_mode:
            self.turn_right(leader)
        elif (selected_action == Action.LEFT) and not leader.left_mode:
            self.turn_left(leader)

        leader.steering = max(-leader.max_steering, min(leader.steering, leader.max_steering))

    def executeActionContinuous(self, selected_action, leader, all_obstacles, dt):
        # TODO: need to test
        # TODO: assume selected_action[0] is acceleration between -1 and 1
        leader.acceleration += selected_action[0] * leader.max_acceleration * dt
        leader.acceleration = max(-leader.max_acceleration, min(leader.acceleration, leader.max_acceleration))

        leader.steering += selected_action[1] * leader.max_steering * dt
        leader.steering = max(-leader.max_steering, min(leader.steering, leader.max_steering))

    # these booleans are required to ensure the action is executed over a period of time
    def accelerate(self, car):
        car.do_accelerate = True
        car.do_decelerate = False
        car.do_maintain = False

    def maintain(self, car, all_obstacles):
        # only execute this function when required
        if car.do_maintain:
            return

        forward_obstacle = None
        for obstacle in all_obstacles:
            if obstacle == car:
                continue
            # obstacle in the same lane
            if obstacle.lane_id == car.lane_id:
                # obstacle has to be ahead the ego vehicle
                if obstacle.position.x > car.position.x:
                    if not forward_obstacle:
                        forward_obstacle = obstacle
                    # obstacle closest to the front
                    elif obstacle.position.x < forward_obstacle.position.x:
                        forward_obstacle = obstacle
        
        obstacle_velx = forward_obstacle.velocity.x if forward_obstacle else car.velocity.x
        car.setCruiseVel(obstacle_velx)
        car.do_maintain = True
        car.do_accelerate = False
        car.do_decelerate = False

    def decelerate(self, car):
        car.do_decelerate = True
        car.do_accelerate = False
        car.do_maintain = False

    def turn_right(self, car):
        car.lane_id = min(car.lane_id + 1, Constants.NUM_LANES)
        car.right_mode = True
        car.left_mode = False

    def turn_left(self, car):
        car.lane_id = max(car.lane_id - 1, 1)
        car.left_mode = True
        car.right_mode = False

    def reset(self):
        if self.render:
            self.bkgd = self.loadBackground()
            self.bkgd_x = 0

        # track data across multiple runs and agents
        self.all_data = [[] for x in range(6)]
        # track data across multiple runs
        self.data_per_run = [[] for x in range(6)]
        self.position_tracker = [[] for x in range(4)]

        # simulation objects
        self.agents = pygame.sprite.Group()
        self.scripted_agents = pygame.sprite.Group()
        self.all_obstacles = pygame.sprite.Group()
        self.lane_max_obs = [None for _ in range(Constants.NUM_LANES)]

        self.reference_car = None

        # set agent objects
        for data in self.policy_agents_data:
            new_agent = Car(id=data['id'], x=data['x'], y=data['y'], vel_x=data['vel_x'], vel_y=data['vel_y'], lane_id=data['lane_id'])
            self.agents.add(new_agent)
            self.all_obstacles.add(new_agent)
            
            if not self.reference_car:
                self.reference_car = new_agent

        for data in self.scripted_agents_data:
            new_obstacle = Obstacle(id=data['id'], x=data['x'], y=data['y'], vel_x=data['vel_x'], vel_y=0.0, lane_id=data['lane_id'], color=data['color'])
            self.scripted_agents.add(new_obstacle)
            self.all_obstacles.add(new_obstacle)
            self.lane_max_obs[new_obstacle.lane_id-1] = new_obstacle

        self.action_timer = 0.0
        self.log_timer = 0.0
        self.continuous_time = 0.0
        # self.current_action = actions.Action.MAINTAIN.name
        self.current_action = actions.Action.ACCELERATE.name
        self.num_obs_collisions = 0
        self.num_agent_collisions = 0

        self.total_velocity_per_run = [self.reference_car.velocity.x]
        self.total_distance_per_run = [self.reference_car.velocity.x]

        self.collision_count_lock = True
        self.run_time = 0.0

        self.is_paused = False
        self.is_done = [False]*len(self.policy_agents_data)

        self.reward = [0.0]*len(self.policy_agents_data)

        return

    def act(self):
        """
            Used by gym-highway to change simulation state only when this function is called

            This needs to be called for 60fps * 0.25s = 15 frames
        """

        # dt = 50.0/1000.0 (For faster simulation)
        # dt = self.clock.get_time() / 1000
        dt = 1.0/self.ticks

        # reset reward so that it corresponds to current action
        self.reward = [0.0]*len(self.policy_agents_data)

        # pause game when needed
        if self.render:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    # self.is_done = True
                    self.close()

        if self.is_paused:
            if self.render:
                pygame.display.flip()
            self.clock.tick(self.framerate)
            return 0.0

        self.action_timer += dt
        self.log_timer += dt
        self.run_time += dt
        self.continuous_time += dt

        if self.run_time >= self.run_duration-(dt/2.0):
            self.is_done = [True]*len(self.policy_agents_data)

        # execute action for each agent (actions are updated in the env after calling update())
        for agent in self.agents:
            self.executeAction(agent, self.all_obstacles, dt)

        # update all sprites
        # this will update all the agents and obstacles based on the actions selected
        self.agents.update(dt, self.reference_car, self.continuous_ctrl)
        self.scripted_agents.update(dt, self.reference_car, self.agents)

        # keep track of agent velocity at end of every action
        if self.log_timer >= Constants.ACTION_RESET_TIME:
            for agent in self.agents:
                self.total_velocity_per_run.append(agent.velocity.x)
            self.log_timer = 0.0

        sorted_agents = sorted(self.agents, key=lambda x: x.position.x, reverse=True)
        self.reference_car = sorted_agents[0]
        # get the last agent to test for obstacle removal
        last_agent = sorted_agents[-1]

        # generate new obstacles
        if self.inf_obstacles:
            # spawn new obstacles when none ahead of lead agent
            for lane in range(Constants.NUM_LANES):
                obstacle = self.lane_max_obs[lane]
                # check the relative position of lead obstacle
                if obstacle.position.x < -Constants.CAR_WIDTH/32:
                    # add new obstacle
                    rand_pos_x = float(random.uniform(70, 100))
                    rand_pos_y = Constants.NEW_LANES[lane]
                    rand_pos_y += random.uniform(-20,20)/Constants.ppu if self.lateral_offset else 0.0
                    rand_vel_x = float(random.uniform(5, 7))
                    rand_lane_id = lane+1

                    # new obs raw position is lead agent raw pos + diff between relative pos
                    new_raw_x = self.reference_car.raw_position.x + (rand_pos_x-self.reference_car.position.x)
                    new_obstacle = Obstacle(id=obstacle.id, x=rand_pos_x, y=rand_pos_y, raw_x=new_raw_x, vel_x=rand_vel_x, vel_y=0.0, lane_id=rand_lane_id, color=Constants.YELLOW)
                    self.scripted_agents.add(new_obstacle)
                    self.all_obstacles.add(new_obstacle)

                    # slow down old obstacle to new obstacle speed
                    adj_min_vel = min(rand_vel_x, obstacle.velocity.x)
                    obstacle.velocity.x = adj_min_vel
                    obstacle.init_velocity.x = adj_min_vel

                    self.lane_max_obs[lane] = new_obstacle

            # remove old obstacle if last agent has cleared it
            # if obs is around -12px behind last agents, remove it
            # obs_pre_count = len(self.scripted_agents)
            # rem_count = 0
            for obstacle in self.scripted_agents:
                # standard overtake distance with a 2px buffer to account for rounding differences in floating point
                overtake_diff = (-Constants.CAR_WIDTH/32 - self.reference_car.position.x) - 2
                if (obstacle.raw_position.x - last_agent.raw_position.x) < overtake_diff:
                    self.scripted_agents.remove(obstacle)
                    self.all_obstacles.remove(obstacle)

                    # TODO: remove after testing
                    # rem_count += 1
            # TODO: check if only appropriate obs are being removed
            # TEST: see if we are removing more than we should
            # assert len(self.scripted_agents) >= 3
            # # TEST: check if remove() deletes more than it should
            # assert len(self.scripted_agents) == obs_pre_count-rem_count
        
        # Drawing
        if self.render:
            self.screen.fill((0, 0, 0))
            
            # Draw The Scrolling Road
            rel_x = self.bkgd_x % self.bkgd.get_rect().width
            self.screen.blit(self.bkgd, (rel_x - self.bkgd.get_rect().width, 0))
            if rel_x < Constants.WIDTH:
                self.screen.blit(self.bkgd, (rel_x, 0))
            self.bkgd_x -= self.reference_car.velocity.x

            # update the agent sprites
            self.updateSprites(self.agents)
            # update obstacle sprites
            self.updateSprites(self.scripted_agents)

            # update collision display count
            self.displayScore(self.num_obs_collisions, self.num_agent_collisions)

            # display velocity of lead car
            self.displayVel(self.reference_car.velocity.x)

            # display position of car
            self.displayPos(self.reference_car.raw_position)

            # display selected action
            self.displayAction(self.reference_car.action)

            pygame.display.flip()

        self.collision_count_lock = False

        self.clock.tick(self.framerate)

    def check_collisions(self):
        # collision check (done before update() to check if previous action led to collisions)
        collisions = [False]*len(self.policy_agents_data)
        if not self.collision_count_lock:
            sorted_agents = sorted(self.agents, key=lambda obj: obj.raw_position.x, reverse=True)
            for i, agent in enumerate(sorted_agents):
                # get collisions with non reactive obstacles
                car_collision_list = pygame.sprite.spritecollide(agent, self.scripted_agents, False)
                obs_collision_val = len(car_collision_list)
                self.num_obs_collisions += obs_collision_val

                # get collisions with reactive agents
                collision_group = self.agents.copy()
                collision_group.remove(agent)
                car_collision_list = pygame.sprite.spritecollide(agent, collision_group, False)
                agent_collision_val = len(car_collision_list)
                self.num_agent_collisions += agent_collision_val

                if (obs_collision_val + agent_collision_val) > 0:
                    collisions[i] = True
                    self.is_done[i] = True
                    # collisions[agent.id] = True
                    # self.is_done[agent.id] = True

        return collisions

    def get_state(self):
        """
            @ Deprecated
            
            FIXME: sprites are not ordered! the observations are randomized each step!

            Observations correspond to positions and velocities of all vehicles on the road.
            Each vehicle contains 4 parameters [pos_x, pos_y, vel_x, vel_y]
            The agent position and velocity is the first entry in the observation space.

            Returns
            -------
            [[pos_x, pos_y, vel_x, vel_y], ... ]
        """
        ref_car = self.reference_car

        ob_list = []
        ob_list.append([ref_car.position.x, ref_car.position.y, ref_car.velocity.x, ref_car.angular_velocity])
        # for idx, obj in enumerate(self.all_obstacles):
        for obj in self.all_obstacles:
            if obj != ref_car:
                ob_list.append([obj.position.x, obj.position.y, obj.velocity.x, obj.velocity.y])

        observations = numpy.array(ob_list, dtype=numpy.float32).flatten()
        return observations

    def get_info(self):
        """
            Info about run time, number of obstacle collitions and number of agent collisions
        """
        info = {"run_time": self.run_time, 
                "num_obs_collisions":self.num_obs_collisions, 
                "num_agent_collisions":self.num_agent_collisions}
        return info

    def is_episode_over(self):
        return self.is_done
    
    def get_done(self):
        return self.is_done

    def close(self):
        pygame.quit()

    def seed(self, seed):
        random.seed(seed)
        numpy.random.seed
