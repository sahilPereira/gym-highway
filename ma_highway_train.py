import argparse
import os
import os.path as osp
import pickle
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import maddpg.common.tf_util as U
import models.config as Config
from baselines import logger
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from models.utils import create_results_dir
from collections import deque

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument('--env', help='environment ID', type=str, default=Config.env_id)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='maddpg')
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=240, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=60000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument('--save_path', help='Location to save trained model', default=None, type=str)
    parser.add_argument('--save_model', default=True, action='store_false')
    parser.add_argument("--log_interval", type=int, default=240, help="timesteps between logging")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=256, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from gym_highway.multiagent_envs.highway_env import MultiAgentEnv
    from gym_highway.multiagent_envs.simple_base import Scenario
    # import multiagent.scenarios as scenarios

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.rewards, scenario.observations, scenario.benchmark_data, scenario.dones)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        # train_step = 0
        t_start = time.time()
        epinfobuf = deque(maxlen=100)

        print('Starting iterations...')
        total_timesteps = arglist.num_episodes*arglist.max_episode_len
        for train_step in range(1, total_timesteps+1):
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = any(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                # save episode info
                epinfobuf.append({"r": episode_rewards[-1], "l":episode_step, "t": round(time.time()-t_start, 6)})
                # reset episode variables
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            if train_step % arglist.log_interval == 0:
                # logger.logkv(Config.tensorboard_rootdir+"serial_timesteps", train_step)
                # logger.logkv(Config.tensorboard_rootdir+"num_update", update)
                logger.logkv(Config.tensorboard_rootdir+"total_timesteps", train_step)
                # logger.logkv(Config.tensorboard_rootdir+"fps", fps)
                # logger.logkv(Config.tensorboard_rootdir+"explained_variance", float(ev))
                logger.logkv(Config.tensorboard_rootdir+'ep_reward_mean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv(Config.tensorboard_rootdir+'ep_length', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv(Config.tensorboard_rootdir+'time_elapsed', round(time.time()-t_start, 6))
                # for (lossval, lossname) in zip(lossvals, model.loss_names):
                #     logger.logkv(Config.tensorboard_rootdir+lossname, lossval)
                if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                    logger.dumpkvs()
            
            # increment global step counter
            # train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model if at save_rate step or if its the first train_step
            save_model = arglist.save_rate and (train_step % arglist.save_rate == 0)
            # only save model if logger dir specified and current node rank is 0 (multithreading)
            save_model &= logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)
            if save_model:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%train_step)
                print('Saving to', savepath)
                U.save_state(savepath, saver=saver)
                # model.save(savepath)
        env.close()

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

if __name__ == '__main__':
    arglist = parse_args()

    # create a separate result dir for each run
    results_dir = create_results_dir(arglist)

    # configure logger
    logger.configure(dir=results_dir, format_strs=Config.baselines_log_format)

    train(arglist)
