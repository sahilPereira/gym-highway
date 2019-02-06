# 3rd party modules
import gym

# internal modules
import gym_highway
from enum import Enum
import pygame
from pygame.math import Vector2
import numpy as np
import tensorflow as tf

import os.path as osp


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def main():
    env = gym.make('Highway-v0')

    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 5
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            # action = 3
            ob, reward, done, _ = env.step(action)
            print("reward: ",reward)
            # print("Ob: ",ob)
            if done:
                print("Done")
                print("Full run reward: ",reward)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

if __name__ == '__main__':
    # pygame.init()
    # pygame.display.set_caption("Car tutorial")
    # width = 1900
    # height = 260
    # screen = pygame.display.set_mode((width, height))
    # clock = pygame.time.Clock()

    # bkgd = pygame.image.load('gym_highway/envs/roadImg.png').convert()
    # print("Img loaded")
    # return bkgd
    # new_arry = np.array([-100,0,0,-25]*4).flatten()
    # print(new_arry)
    # flattened_array = new_arry.flatten()
    # print(flattened_array)

    # myList = [0.30000000000000004, 0.5, 0.20000000000000001]
    # myRoundedList = [ round(elem, 2) for elem in myList ]
    # print(myRoundedList)

    # file_path = "~/baselines/test1/"
    # save_path = osp.expanduser(file_path)
    # print(save_path)
    obs = np.asarray([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
    returns = np.asarray([11,12,13,14,15,16,17,18,19,20], dtype=np.float32)
    masks = np.asarray([21,22,23,24,25,26,27,28,29,30], dtype=np.float32)
    # inds = np.arange(nbatch)
    mbinds = np.array([4,5,6,7,8])

    slices = [arr[mbinds] for arr in (obs, returns, masks)]

    # *map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs))
    # slices = np.array(slices)
    # print(slices)
    # print(slices)

    # epbonuses = [0.0]*5

    epbonuses = np.zeros(5, dtype=np.float32)
    # epbonuses = np.add(epbonuses, [2,4,1,5,6])
    epbonuses += [2,4,1,5,6]
    # epbonuses = [x + y for x, y in zip(epbonuses, [2,4,1,5,6])]
    print(epbonuses)

    epbonuses += [1,1,1,1,1]
    # epbonuses = [x + y for x, y in zip(epbonuses, [1,1,1,1,1])]
    # epbonuses = np.add(epbonuses, [1,1,1,1,1])
    print(epbonuses)

    epbonuses[1] = 0.0
    epbonuses[3] = 0.0
    print(epbonuses)

    epbonuses += [5,5,3,2,1]
    # epbonuses = np.add(epbonuses, [5,5,3,2,1])
    print(epbonuses)

    epbonuses = epbonuses*2.0
    print(epbonuses)
    
    # x = tf.placeholder(shape=(None, 16), dtype='float32')
    # y = tf.layers.flatten(x)

    # sess = tf.Session()
    # x_val = np.random.rand(37, 16)
    # y_out = sess.run(y, {x: x_val})

    # print(y_out.shape)
    
    # main()