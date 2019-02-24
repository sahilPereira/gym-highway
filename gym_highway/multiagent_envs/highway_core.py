import math
import os
import pickle
import random
from argparse import ArgumentParser
from enum import Enum
from math import ceil, copysign, degrees, radians, tan
from random import randrange

import numpy
import pandas as pd
import pygame
from pygame.math import Vector2

from gym_highway.envs.stackelbergPlayer import Action, StackelbergPlayer
import gym_highway.multiagent_envs.highway_constants as Constants

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    ACCELERATE = 2
    MAINTAIN = 3
    # DECELERATE = 3

class HighwaySimulator:
    def __init__(self, cars_list, obstacle_list, is_manual=False, inf_obstacles=False, is_data_saved=False, render=False):
        pygame.init()
        width = Constants.WIDTH
        height = Constants.HEIGHT
        if render:
            pygame.display.set_caption("Car tutorial")
            self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

        # simulation options
        self.cars_list = cars_list
        self.obstacle_list = obstacle_list
        self.is_manual = is_manual
        self.inf_obstacles = inf_obstacles
        self.is_data_saved = is_data_saved
        self.render = render
        self.total_runs = 100
        self.run_duration = 60 # 60 seconds

        # log files
        self.files_dir = 'datafiles/NoDelay/dt_0.05/'
        self.file_name = self.files_dir+'model_IGA_0.05_%d_%d_%d.csv'%(len(self.cars_list), self.total_runs, self.run_duration)

        self.df_columns = ['agent_count','run_count','obs_collisions','agent_collisions', 'avg_velocity', 'avg_distance']
        self.position_columns = ['agent_id','time','pos_x','pos_y']

        # reset simulator states
        self.reset()

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

    def displayPos(self, position):
        font = pygame.font.SysFont(None, 25)
        # text = font.render("X: "+str(position.x)+", Y: "+str(position.y), True, WHITE)
        text = font.render("Velocity: "+str(position), True, Constants.WHITE)
        self.screen.blit(text, (0,50))

    def displayAction(self, action):
        font = pygame.font.SysFont(None, 25)
        text = font.render("Action: "+str(action), True, Constants.WHITE)
        self.screen.blit(text, (0,75))

    def updateSprites(self, vehicles):
        for auto in vehicles:
            rotated = pygame.transform.rotate(auto.image, auto.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, auto.position * Constants.ppu - (rect.width / 2, rect.height / 2))

    def manualControl(self, car, all_obstacles):
        # User input
        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_UP] and not car.do_accelerate:
            self.accelerate(car)
        elif pressed[pygame.K_DOWN] and not car.do_maintain:
            self.maintain(car, all_obstacles)
        elif pressed[pygame.K_SPACE] and not car.do_decelerate:
            self.decelerate(car)

        car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))

        if pressed[pygame.K_RIGHT] and not car.right_mode:
            self.turn_right(car)
        elif pressed[pygame.K_LEFT] and not car.left_mode:
            self.turn_left(car)

        car.steering = max(-car.max_steering, min(car.steering, car.max_steering))

    def stackelbergControl(self, controller, reference_car, all_agents, all_obstacles):

        # Step 1. select players to execute action at this instance
        players = controller.pickLeadersAndFollowers(all_agents, all_obstacles)

        # Step 2. iterate over the set of players and execute their actions
        for leader in players:
            # Step 3: select actions for all players from step 2 sequentially
            selected_action = controller.selectAction(leader, all_obstacles)

            # select action using Stackelberg game
            # selected_action = controller.selectStackelbergAction(leader, all_obstacles, reference_car)

            self.executeAction(selected_action, leader, all_obstacles)

        # Note that every player acts as a leader when selecting their actions
        return selected_action

    # execute the given action for the specified leader
    def executeAction(self, selected_action, leader, all_obstacles):
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
        
        return

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
        self.all_agents = pygame.sprite.Group()
        self.all_obstacles = pygame.sprite.Group()
        self.all_coming_cars = pygame.sprite.Group()

        self.reference_car = None
        for data in self.cars_list:
            new_car = Car(id=data['id'], x=data['x'], y=data['y'], vel_x=data['vel_x'], vel_y=data['vel_y'], lane_id=data['lane_id'])
            self.all_agents.add(new_car)
            self.all_obstacles.add(new_car)

            if not self.reference_car:
                self.reference_car = new_car

        for data in self.obstacle_list:
            new_obstacle = Obstacle(id=data['id'], x=data['x'], y=data['y'], vel_x=data['vel_x'], vel_y=0.0, lane_id=data['lane_id'], color=data['color'])
            self.all_coming_cars.add(new_obstacle)
            self.all_obstacles.add(new_obstacle)

        self.action_timer = 0.0
        self.log_timer = 0.0
        self.continuous_time = 0.0
        # self.current_action = Action.MAINTAIN.name
        self.current_action = Action.ACCELERATE.name
        self.num_obs_collisions = 0
        self.num_agent_collisions = 0

        self.total_velocity_per_run = [self.reference_car.velocity.x]
        self.total_distance_per_run = [self.reference_car.velocity.x]

        self.collision_count_lock = True
        self.run_time = 0.0

        self.is_paused = False
        self.is_done = False

        self.reward = 0.0

        return

    def act(self, action):
        """
            Used by gym-highway to change simulation state only when this function is called

            This needs to be called for 60fps * 0.25s = 15 frames
        """

        # dt = 50.0/1000.0 (For faster simulation)
        dt = self.clock.get_time() / 1000

        # reset reward so that it corresponds to current action
        self.reward = 0.0

        # pause game when needed
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # self.is_done = True
                self.close()
            # if e.type == pygame.KEYDOWN:
            #     if e.key == pygame.K_p: self.is_paused = True
            #     if e.key == pygame.K_r: self.is_paused = False

        if self.is_paused:
            if self.render:
                pygame.display.flip()
            self.clock.tick(self.ticks)
            return 0.0

        self.action_timer += dt
        self.log_timer += dt
        self.run_time += dt
        self.continuous_time += dt

        self.is_done = self.run_time >= self.run_duration

        # workign with just one agent for now
        # TODO: need to change this when working with multiple agents
        self.executeAction(action, self.reference_car, self.all_obstacles)

        # collision check
        if not self.collision_count_lock:
            for agent in self.all_agents:
                # get collisions with non reactive obstacles
                car_collision_list = pygame.sprite.spritecollide(agent,self.all_coming_cars,False)
                obs_collision_val = len(car_collision_list)
                self.num_obs_collisions += obs_collision_val

                collision_group = self.all_agents.copy()
                collision_group.remove(agent)
                car_collision_list = pygame.sprite.spritecollide(agent,collision_group,False)
                agent_collision_val = len(car_collision_list)
                self.num_agent_collisions += agent_collision_val

                # TODO: changed the crash multiplier to 10.0 instead of 1.0
                self.reward -= (obs_collision_val + agent_collision_val)*250.0
                if self.reward < 0:
                    # end run when crash occurs
                    self.is_done = True

        # update all sprites
        self.all_agents.update(dt, self.reference_car)
        self.all_coming_cars.update(dt, self.reference_car)

        # max reward per step is 1.0 for going at max velocity
        if self.reward == 0.0:
            # self.reward = self.reference_car.velocity.x / self.reference_car.max_velocity

            # TODO: reward of 0 for going max speed, negative reward otherwise
            self.reward = (self.reference_car.velocity.x / self.reference_car.max_velocity) - 1.0

            # TODO: test delayed reward
            # if self.is_done:
            #     self.reward = (sum(self.total_velocity_per_run) / float(len(self.total_velocity_per_run))) / self.reference_car.max_velocity

        # keep track of agent velocity at end of every action
        if self.log_timer >= Constants.ACTION_RESET_TIME:
            for agent in self.all_agents:
                self.total_velocity_per_run.append(agent.velocity.x)
            self.log_timer = 0.0

        sorted_agents = sorted(self.all_agents, key=lambda x: x.position.x, reverse=True)
        self.reference_car = sorted_agents[0]

        # generate new obstacles
        if self.inf_obstacles:
            for obstacle in self.all_coming_cars:
                if obstacle.position.x < -Constants.CAR_WIDTH/32:
                    # remove old obstacle
                    self.all_coming_cars.remove(obstacle)
                    self.all_obstacles.remove(obstacle)
                    # obstacle_lanes.remove(obstacle.lane_id)

                    # add new obstacle
                    rand_pos_x = float(randrange(70, 80))
                    rand_pos_y = Constants.NEW_LANES[obstacle.lane_id-1]
                    rand_vel_x = float(randrange(5, 15))
                    rand_lane_id = obstacle.lane_id

                    new_obstacle = Obstacle(id=randrange(100,1000), x=rand_pos_x, y=rand_pos_y, vel_x=rand_vel_x, vel_y=0.0, lane_id=rand_lane_id, color=Constants.YELLOW)
                    self.all_coming_cars.add(new_obstacle)
                    self.all_obstacles.add(new_obstacle)

                    # TEST: reward for overtaking each vehicle
                    # self.reward += 1

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
            self.updateSprites(self.all_agents)
            # update obstacle sprites
            self.updateSprites(self.all_coming_cars)

            # update collision display count
            self.displayScore(self.num_obs_collisions, self.num_agent_collisions)

            # display position of car
            self.displayPos(self.reference_car.velocity.x)

            # display selected action
            self.displayAction(action)

            pygame.display.flip()

        self.collision_count_lock = False

        self.clock.tick(self.ticks)
        # self.clock.tick()

        return self.reward

    def get_state(self):
        """
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

    def close(self):
        pygame.quit()

    def seed(self, seed):
        random.seed(seed)
        numpy.random.seed
