"""Example of a custom gym environment. Run this for a demo."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym_highway.envs import HighwayEnv

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
# from ray.rllib.agents.ddpg.ddpg_policy_graph import DDPGPolicyGraph
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
# from ray.rllib.agents.ddpg import DDPGAgent
from ray.rllib.agents.ppo import PPOAgent

def trainGymHighway():
    env_creator_name = "gym_highway"
    register_env(env_creator_name, lambda _: HighwayEnv(False, True, False, False))
    ray.init()

    ppo_agent = PPOAgent(
        env=env_creator_name,
        config={
            "num_workers": 2,
            "num_envs_per_worker": 1,
            "sample_batch_size":50,
            "train_batch_size":1000,
            "num_gpus":1,
        })

    # Just check that it runs without crashing
    best_reward = 0
    # TODO: testing for 1 hour (60 runs)
    for i in range(60):
        result = ppo_agent.train()

        print("Iteration {}, reward {}, timesteps {}".format(
            i, result["episode_reward_mean"], result["timesteps_total"]))

        if result["episode_reward_mean"] > best_reward:
            checkpoint = ppo_agent.save()
            print("checkpoint saved at", checkpoint)
            best_reward = result["episode_reward_mean"]

if __name__ == "__main__":
    
    trainGymHighway()
    print("Done Training")
