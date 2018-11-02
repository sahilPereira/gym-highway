"""Example of a custom gym environment. Run this for a demo."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import gym
import gym_highway
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym_highway.envs import HighwayEnv
from argparse import ArgumentParser

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
# from ray.rllib.agents.ddpg.ddpg_policy_graph import DDPGPolicyGraph
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
# from ray.rllib.agents.ddpg import DDPGAgent
from ray.rllib.agents.ppo import PPOAgent

# import argparse
import json
import os
# import pickle

# import gym
# import ray
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog

def trainGymHighway():
    """
    TODO: 
        - check gamma (discount factor is around 0.95)
        - normalize rewards
        - double check if rewards are being added properly (Fixed: separate reward for each step)
        - check policy models
        - check activations
    """

    env_creator_name = "Highway-v0"
    register_env(env_creator_name, lambda _: HighwayEnv(False, True, False, False))
    ray.init()

    ppo_agent = PPOAgent(
        env=env_creator_name,
        config={
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "sample_batch_size":64,
            "train_batch_size":1280,
            "num_gpus":1,
        })

    # TODO: restore last best checkpoint
    # checkpoint = "/home/s6pereir/ray_results/PPO_Highway-v0_2018-10-29_19-26-438_endbsi/checkpoint-100"
    # ppo_agent.restore(checkpoint)

    # Just check that it runs without crashing
    best_reward = 0
    # TODO: testing for 1 hour (60 runs)
    for i in range(1000):
        result = ppo_agent.train()

        print("Iteration {}, reward {}, timesteps {}".format(
            i, result["episode_reward_mean"], result["timesteps_total"]))

        if result["episode_reward_mean"] > best_reward:
            checkpoint = ppo_agent.save()
            print("checkpoint saved at", checkpoint)
            best_reward = result["episode_reward_mean"]


# rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    # --env CartPole-v0 --steps 1000000 --out rollouts.pkl
# https://ray.readthedocs.io/en/latest/rllib-training.html#python-api
def testGymHighway(args):
    checkpoint_dir = "/home/s6pereir/ray_results/PPO_Highway-v0_2018-10-30_00-14-29nniwg382/"
    checkpoint = "/home/s6pereir/ray_results/PPO_Highway-v0_2018-10-30_00-14-29nniwg382/checkpoint-101"

    # Load configuration from file
    config_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(config_dir, "params.json")
    with open(config_path) as f:
        # args.config = json.load(f)
        config = json.load(f)

    env_creator_name = "Highway-v0"
    register_env(env_creator_name, lambda _: HighwayEnv(False, True, False, True))
    env = gym.make('Highway-v0')

    ray.init()

    num_steps = int(10)
    agent = PPOAgent(env=env_creator_name)
    agent.restore(checkpoint)
    
    steps = 0
    while steps < num_steps:
        state = env.reset()
        reward_total = 0.0
        while True:
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            state = next_state
            if done:
                print("Done")
                break
        steps += 1
        print("Episode reward", reward_total)
    env.close()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--train", dest="train", action='store_true', help="Train model")
    parser.add_argument("--test", dest="test", action='store_true', help="Test model with visualization")
    
    args = parser.parse_args()

    if args.train:
        print("Training Model")
        trainGymHighway()
        print("Done Training")
    elif args.test:
        print("Testing Model")
        testGymHighway({"config":None,})
    else:
        print("Specify --train or --test")
