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

from gym_highway.multiagent_envs import highway_constants as Constants
from gym_highway.multiagent_envs import actions

class Car(pygame.sprite.Sprite):
    def __init__(self, id, x, y, vel_x=0.0, vel_y=0.0, lane_id=1, color=Constants.RED, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):

        # init the sprite object
        super().__init__()

        self.id = id
        self.image = pygame.Surface([Constants.CAR_WIDTH, Constants.CAR_HEIGHT])
        self.image.fill(Constants.WHITE)
        self.image.set_colorkey(Constants.WHITE)
 
        # Draw the car (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, Constants.CAR_WIDTH, Constants.CAR_HEIGHT])
 
        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
        self.position = Vector2(x, y)
        self.velocity = Vector2(vel_x, vel_y)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20.0
        self.brake_deceleration = 10.0
        self.free_deceleration = 2.0
        self.acceleration = 0.0
        self.steering = 0.0
        self.angular_velocity = 0.0
        self.lane_id = lane_id
        self.left_mode, self.right_mode, self.do_accelerate, self.do_decelerate, self.do_maintain = False, False, False, False, False
        self.cruise_vel = 0.0

        # action
        self.action = actions.Action.MAINTAIN
        # script behavior to execute
        self.action_callback = None

    def simCopy(self):
        sim_car = Car(self.id, self.position.x, self.position.y, self.velocity.x, self.velocity.y, self.lane_id)
        # dynamic controls
        sim_car.acceleration = self.acceleration
        sim_car.steering = self.steering
        # action controls
        sim_car.left_mode = self.left_mode
        sim_car.right_mode = self.right_mode
        sim_car.do_accelerate = self.do_accelerate
        sim_car.do_decelerate = self.do_decelerate
        sim_car.do_maintain = self.do_maintain
        sim_car.cruise_vel = self.cruise_vel

        return sim_car

    def update(self, dt, s_leader):
        
        if self.do_accelerate:
            self.accelerate(dt)
        elif self.do_decelerate:
            self.decelerate(dt)
        elif self.do_maintain:
            self.maintain(dt)

        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(0.0, min(self.velocity.x, self.max_velocity))

        # trigger movement
        new_lane_pos = (Constants.LANE_WIDTH * self.lane_id - (Constants.LANE_WIDTH/2))/Constants.ppu
        # print(new_lane_pos)
        if self.left_mode:
            self.moveLeft(dt, new_lane_pos)
        elif self.right_mode:
            self.moveRight(dt, new_lane_pos)

        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            self.angular_velocity = self.velocity.x / turning_radius
        else:
            self.angular_velocity = 0
        # update y component as well
        self.velocity.y = self.angular_velocity

        self.position += self.velocity.rotate(-self.angle) * dt
        self.position.y -= degrees(self.angular_velocity) * dt * dt

        if self.id == s_leader.id:
            self.position.x = 10
        else:
            self.position.x -= s_leader.velocity.x * dt

        # prevent the car from leaving the road
        if self.position.y < int((Constants.CAR_HEIGHT/2)/Constants.ppu):
            self.position.y = max(self.position.y, int((Constants.CAR_HEIGHT/2)/Constants.ppu))
        elif self.position.y > int((Constants.HEIGHT - int(Constants.CAR_HEIGHT/2))/Constants.ppu):
            self.position.y = min(self.position.y, int((Constants.HEIGHT - int((Constants.CAR_HEIGHT/2)/Constants.ppu))/Constants.ppu))

        # update rect for collision detection
        self.rect.x = self.position.x * Constants.ppu - self.rect.width / 2
        self.rect.y = self.position.y * Constants.ppu - self.rect.height / 2

    def setCruiseVel(self, cruise_vel):
        self.cruise_vel = cruise_vel

    def moveLeft(self, dt, new_lane_pos):
        self.steering += 30.0 * dt

        if self.position.y <= new_lane_pos:
            self.steering = 0
            self.left_mode = False

    def moveRight(self, dt, new_lane_pos):
        self.steering -= 30.0 * dt

        if self.position.y >= new_lane_pos:
            self.steering = 0
            self.right_mode = False

    def accelerate(self, dt):
        # the longitudinal velocity should never be less than 0
        if self.acceleration < 0.0:
            self.acceleration = 0.0
        else:
            self.acceleration += 1 * dt
        if self.acceleration == self.max_acceleration:
            self.do_accelerate = False

    def maintain(self, dt):
        vel_ceil = ceil(self.velocity.x)
        cruise_vel_ceil = ceil(self.cruise_vel)

        # check if car needs to speed up or slow down and accelerate accordingly
        is_speed = True if vel_ceil <= cruise_vel_ceil else False
        self.acceleration = self.max_acceleration if is_speed else -self.max_acceleration

        # speed up or slow down until the car reaches cruise velocity
        is_cruise_speed = is_speed and vel_ceil >= cruise_vel_ceil
        is_cruise_speed |= (not is_speed) and vel_ceil <= cruise_vel_ceil

        if is_cruise_speed:
            self.velocity.x = self.cruise_vel
            self.acceleration = 0.0
            self.do_maintain = False

    def decelerate(self, dt):
        if self.acceleration > 0.0:
            self.acceleration = -self.max_acceleration #0.0
        else:
            self.acceleration -= 1 * dt
        if self.velocity.x == 0.0:
            self.do_decelerate = False

class Obstacle(Car):
    def __init__(self, *args, **kw):
        # init the sprite object
        super().__init__(*args, **kw)
