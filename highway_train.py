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

import json
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import normc_initializer, get_activation_fn


class CustomFCModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.
        """
        hiddens = options.get("fcnet_hiddens", [256, 256, 256])
        activation = get_activation_fn(options.get("fcnet_activation", "relu"))

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    scope=label)
                i += 1
            label = "fc_out"
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope=label)
            return output, last_layer

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return tf.reshape(
            linear(self.last_layer, 1, "value", normc_initializer(1.0)), [-1])

def register_custom_model():
    ModelCatalog.register_custom_model("custom_fc_model", CustomFCModel)

def trainGymHighway():
    """
    TODO: 
        - check gamma (discount factor is around 0.95)
        - normalize rewards
        - double check if rewards are being added properly (Fixed: separate reward for each step)
        - check policy models
        - check activations
        - Consider using a LSTM since it can help with making decisions over time
            - I should definitely be using LSTMs here since they will help us see how our actions affect our state over time
            - Right now we are only making decision based on the current state without considering the history.
    """

    env_creator_name = "Highway-v0"
    register_env(env_creator_name, lambda _: HighwayEnv(False, True, False, False))

    # register custom model
    register_custom_model()

    ray.init(num_gpus=1)

    ppo_agent = PPOAgent(
        env=env_creator_name,
        config={
            "num_workers": 4,
            "num_envs_per_worker": 1,
            # "sample_batch_size":64,
            # "train_batch_size":1280,
            "num_gpus":1,
            "model": {
                "custom_model": "custom_fc_model",
                "custom_options": {},
            },
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
