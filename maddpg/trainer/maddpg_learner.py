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
from gym import spaces
from gym.envs.registration import register

import maddpg.common.tf_util as U
import models.config as Config
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.cmd_util import (common_arg_parser, make_env,
                                       make_vec_env, parse_unknown_args)
from baselines.common.models import get_network_builder
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from models.utils import (activation_str_function, create_results_dir,
                          parse_cmdline_kwargs, save_configs)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    # TODO: remove after testing
    assert num_layers == Config.maddpg_train_args['num_layers']
    assert num_hidden == Config.maddpg_train_args['num_hidden']
    assert activation == tf.nn.relu

    if isinstance(num_hidden, int):
        num_hidden = [num_hidden]*num_layers

    def network_fn(X):
        h = X
        for i in range(num_layers):
            h = layers.fully_connected(h, num_outputs=num_hidden[i], activation_fn=activation)
        return h
    return network_fn

def create_model(**network_kwargs):
    # create mlp model using custom args
    mlp_network_fn = mlp(**network_kwargs)

    # This model takes as input an observation and returns values of all actions
    def mlp_model(input, num_outputs, scope, reuse=False, num_units=256, rnn_cell=None):
        with tf.variable_scope(scope, reuse=reuse):
            out = mlp_network_fn(input)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return out
    return mlp_model

def get_trainers(env, num_agents, num_adversaries, obs_shape_n, adv_policy, good_policy, training_params, **network_kwargs):
    trainers = []
    model = create_model(**network_kwargs)
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, {}, **training_params,
            local_q_func=(adv_policy=='ddpg')))
    for i in range(num_adversaries, num_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, {}, **training_params,
            local_q_func=(good_policy=='ddpg')))
    return trainers

def learn(env,
          total_timesteps,
          num_agents=1,
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
          adv_policy='maddpg',
          good_policy='maddpg',
          load_path=None,
          save_interval=100,
          num_adversaries=0,
          rb_size=1e6,
          **network_kwargs):
    
    set_global_seeds(seed)

    continuous_ctrl = not isinstance(env.action_space, spaces.Discrete)
    nb_actions = env.action_space[0].shape[-1] if continuous_ctrl else env.action_space[0].n

    # update training parameters
    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    # 1. Create agent trainers
    # replay buffer, actor and critic are defined for each agent in trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(num_agents)]
    num_adversaries = min(num_agents, num_adversaries)
    training_params = {'actor_lr':actor_lr, 'critic_lr':critic_lr, 'gamma':gamma, 
                       'num_units':network_kwargs['num_hidden'], 'rb_size':rb_size, 
                       'batch_size':batch_size, 'max_episode_len':nb_rollout_steps, 
                       'clip_norm':clip_norm}

    trainers = get_trainers(env, num_agents, num_adversaries, obs_shape_n, adv_policy, good_policy, training_params, **network_kwargs)
    print("Num of observations: {}".format(len(obs_shape_n)))
    print('Observation shapes {}'.format(obs_shape_n))

    # 2. define parameter and action noise
    # not done in maddpg, but done in ddpg

    # 3. define action scaling
    # not done in maddpg, but done in ddpg

    # 4. define agent(s)
    # already done defining maddpg models in get_trainers

    # 5. output any useful logging information
    # logger.info('scaling actions by {} before executing in env'.format(max_action))

    
    # 6. get session and initialize all agent variables
    # TODO: might just need to use get_session() if sess already created
    with U.single_threaded_session():
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        # TODO: might need to update this based on how we save model
        if load_path is not None:
            print('Loading previous state...')
            U.load_state(load_path)

        obs_n = env.reset()
        nenvs = obs_n.shape[0]
        # ensure the shape of obs is consistent
        assert obs_n.shape == (nenvs, num_agents, obs_n.shape[-1])

        # 8. initialize metric tracking parameters
        episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
        episode_step = np.zeros(nenvs, dtype = int) # vector
        episodes = 0 #scalar
        t = 0 # scalar
        epoch = 0

        start_time = time.time()

        epoch_episode_rewards = 0.0
        epoch_episode_steps = 0.0
        epoch_actions = 0.0
        epoch_qs = []
        epoch_episodes = 0

        # training metrics
        loss_metrics = {'q_loss':deque(maxlen=len(trainers)), 
                        'p_loss':deque(maxlen=len(trainers)), 
                        'mean_target_q':deque(maxlen=len(trainers)), 
                        'mean_rew':deque(maxlen=len(trainers)), 
                        'mean_target_q_next':deque(maxlen=len(trainers)), 
                        'std_target_q':deque(maxlen=len(trainers))
                       }

        saver = tf.train.Saver()
        episode_rewards_history = deque(maxlen=100)

        # 9. nested training loop
        print('Starting iterations...')
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):

                # 7. reset agents and envs
                # NOTE: since we dont have action and param noise, no agent.reset() required here
                 
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    actions_n = []
                    for i in range(nenvs):
                        # get actions for all agents in current env
                        actions_n.append([agent.action(obs) for agent, obs in zip(trainers,obs_n[i])])
                        
                    # confirm actions_n is nenvs x num_agents x len(Action)
                    assert np.array(actions_n).shape == (nenvs, num_agents, nb_actions)
                    
                    # environment step
                    new_obs_n, rew_n, done_n, info_n = env.step(actions_n)

                    # sum of rewards for each env
                    episode_reward += [sum(r) for r in rew_n]
                    episode_step += 1

                    # Book-keeping
                    for i, agent in enumerate(trainers):
                        for b in range(nenvs):
                            # save experience from all envs for each agent
                            agent.experience(obs_n[b][i], actions_n[b][i], rew_n[b][i], new_obs_n[b][i], done_n[b][i], None)
                    obs_n = new_obs_n

                    for d in range(len(done_n)):
                        if any(done_n[d]):
                            # Episode done.
                            epoch_episode_rewards += episode_reward[d]
                            episode_rewards_history.append(episode_reward[d])
                            epoch_episode_steps += episode_step[d]
                            episode_reward[d] = 0.
                            episode_step[d] = 0
                            epoch_episodes += 1
                            episodes += 1
                            # if nenvs == 1:
                            #     agent.reset()
                    
                    # update timestep
                    t += 1
                
                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []

                for t_train in range(nb_train_steps):
                    for agent in trainers:
                        agent.preupdate()
                        loss = agent.update(trainers, t)
                        # continue if there is no loss computed
                        if not loss:
                            continue

                        # get all the loss metrics for this agent
                        lossvals = [np.mean(data, axis=0) if isinstance(data, list) else data for data in loss]
                        # add the metrics to respective queue
                        for (lossval, lossname) in zip(lossvals, agent.loss_names):
                            loss_metrics[lossname].append(lossval)
                
                # TODO: implement evaluate logic (not included here)

            # 10. logging metrics
            duration = time.time() - start_time
            combined_stats = {}
            combined_stats[Config.tensorboard_rootdir+'rollout/return'] = epoch_episode_rewards / float(episodes)
            combined_stats[Config.tensorboard_rootdir+'rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats[Config.tensorboard_rootdir+'rollout/episode_steps'] = epoch_episode_steps / float(episodes)
            combined_stats[Config.tensorboard_rootdir+'train/loss_actor'] = np.mean(loss_metrics['p_loss'])
            combined_stats[Config.tensorboard_rootdir+'train/loss_critic'] = np.mean(loss_metrics['q_loss'])
            combined_stats[Config.tensorboard_rootdir+'train/mean_target_q'] = np.mean(loss_metrics['mean_target_q'])
            combined_stats[Config.tensorboard_rootdir+'train/mean_rew'] = np.mean(loss_metrics['mean_rew'])
            combined_stats[Config.tensorboard_rootdir+'train/mean_target_q_next'] = np.mean(loss_metrics['mean_target_q_next'])
            combined_stats[Config.tensorboard_rootdir+'train/std_target_q'] = np.mean(loss_metrics['std_target_q'])
            combined_stats[Config.tensorboard_rootdir+'total/duration'] = duration
            combined_stats[Config.tensorboard_rootdir+'total/steps_per_second'] = float(t) / float(duration)
            combined_stats[Config.tensorboard_rootdir+'total/episodes'] = episodes
            combined_stats[Config.tensorboard_rootdir+'rollout/episodes'] = epoch_episodes
            
            # Evaluation statistics.
            # if eval_env is not None:
            #     combined_stats[Config.tensorboard_rootdir+'eval/return'] = eval_episode_rewards
            #     combined_stats[Config.tensorboard_rootdir+'eval/return_history'] = np.mean(eval_episode_rewards_history)
            #     combined_stats[Config.tensorboard_rootdir+'eval/Q'] = eval_qs
            #     combined_stats[Config.tensorboard_rootdir+'eval/episodes'] = len(eval_episode_rewards)


            combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
            if MPI is not None:
                combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

            mpi_size = MPI.COMM_WORLD.Get_size() if MPI is not None else 1
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats[Config.tensorboard_rootdir+'total/epochs'] = epoch + 1
            combined_stats[Config.tensorboard_rootdir+'total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])

            if rank == 0:
                logger.dump_tabular()
            logger.info('')

            # 11. saving model when required
            if save_interval and (epoch % save_interval == 0) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%epoch)
                print('Saving to', savepath)
                # TODO: test which method works
                # agent.save(savepath)
                U.save_state(savepath, saver=saver)
            
    return trainers
