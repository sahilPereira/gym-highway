import argparse
import multiprocessing
import os
import os.path as osp
import pickle
import sys
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gym.envs.registration import register

import maddpg.common.tf_util as U
import models.config as Config
from baselines import logger
from baselines.common.cmd_util import (common_arg_parser, make_env,
                                       make_vec_env, parse_unknown_args)
from baselines.common.tf_util import get_session
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from models.utils import (activation_str_function, create_results_dir,
                          parse_cmdline_kwargs, save_configs)
from baselines.common import set_global_seeds
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    # TODO: update this to work with standard model generator
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

def learn(env, 
          total_timesteps, 
          arglist,
          seed=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          save_interval=100,
          num_adversaries=0,
          **network_kwargs):
    
    set_global_seeds(seed)

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    # 1. Create agent trainers
    # replay buffer, actor and critic are defined for each agent in trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    # 2. define parameter and action noise
    # not done in maddpg, but done in ddpg

    # 3. define action scaling
    # not done in maddpg, but done in ddpg

    # 4. define agent(s)
    # already done defining maddpg models in get_trainers

    # 5. output any useful logging information

    # 6. get session and initialize all agent variables

    # 7. reset agents and envs

    # 8. initialize metric tracking parameters

    # 9. nested training loop

    # 10. logging metrics

    # 11. saving model when required
    
    with U.single_threaded_session():
        # Create environment
        # env = make_env(arglist.scenario, arglist, arglist.benchmark)

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
                logger.logkv(Config.tensorboard_rootdir+"current_episode", int(train_step/arglist.max_episode_len))
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

                # log loss info for each agent
                if (train_step % arglist.log_interval == 0) and loss:
                    lossvals = [np.mean(data, axis=0) if isinstance(data, list) else data for data in loss]
                    for (lossval, lossname) in zip(lossvals, agent.loss_names):
                        log_key = "{}{}/{}".format(Config.tensorboard_rootdir, lossname, agent.name)
                        logger.logkv(log_key, lossval)

            # save model if at save_rate step or if its the first train_step
            save_model = arglist.save_rate and ((train_step % arglist.save_rate == 0) or (train_step == total_timesteps))
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