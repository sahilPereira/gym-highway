import argparse
import multiprocessing
import os
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict, deque

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
# from maddpg.trainer.maddpg_learner import learn
from maddpg.trainer.ma_ddpg import learn
from models.utils import (activation_str_function, create_results_dir,
                          parse_cmdline_kwargs, save_configs)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_game_envs = defaultdict(set)

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def parse_args():
    parser = arg_parser()
    # Environment
    parser.add_argument('--env', help='environment ID', type=str, default=Config.ma_c_env_id)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--branch', help='current git branch', type=str, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='maddpg')
    parser.add_argument('--scenario', help='Scenario', type=str, default='simple_spread')
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e6)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel', default=Config.num_workers, type=int)
    parser.add_argument("--num_agents", type=int, default=1, help="number of total agents")
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--continuous', default=False, help='Use continuous actions', action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    return parser

def mlp_model(input, num_outputs, scope, reuse=False, num_units=256, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env_config(arglist, benchmark=False):
    # get world config
    world_config = Config.env_play_kwargs if arglist.play else Config.env_train_kwargs
    # return multiagent environment config
    return {'world_config':world_config, 'num_agents':arglist.num_agents, 'shared_reward':False}

def make_env(scenario_name, arglist, benchmark=False):
    # return {'scenario_name':scenario_name}
    from gym_highway.multiagent_envs.multiagent.environment import MultiAgentEnv
    import gym_highway.multiagent_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
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

def train(args, extra_args):
    # env_type, env_id = get_env_type(args.env)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    # TODO: not required right now, might need in future to make learn function more modular
    alg_kwargs = {}
    # learn = get_learn_function(args.alg)
    # alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    # env = build_env(args)
    env = make_env(args.scenario, args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        total_timesteps=total_timesteps,
        num_agents=args.num_agents,
        seed=seed,
        **alg_kwargs
    )
    return model, env

def build_env(args):
    '''
    Build a vector of n environments
    '''
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her', 'maddpg'}
    env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations, isMultiAgent=True)

    return env

def get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def learn_old(env, 
          arglist,
          seed=None,
          total_timesteps=None,
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
          **network_kwargs):
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

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def register_env(c_id, c_entry_point, c_kwargs):
    '''
    Register a gym environment with new id, entry point and kwargs
    '''
    register( id=c_id, entry_point=c_entry_point, kwargs=c_kwargs)

if __name__ == '__main__':
    args = sys.argv
    arg_parser = parse_args()
    args, unknown_args = arg_parser.parse_known_args(args)

    # update env being used based on action type
    args.env = Config.ma_c_env_id if args.continuous else Config.ma_env_id
    
    # add custom training arguments for ppo2 algorithm
    # TODO: update to MA specific activation functions
    extra_args = Config.maddpg_train_args
    extra_args = activation_str_function(extra_args)

    # update extra_args with command line argument overrides
    extra_args.update(parse_cmdline_kwargs(unknown_args))

    # create a separate result dir for each run
    results_dir = create_results_dir(args)

    print("-----> Results saved to: {}".format(results_dir))

    # save configurations
    save_configs(results_dir, args, extra_args)

    # configure logger
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=results_dir, format_strs=Config.baselines_log_format)
    else:
        logger.configure(dir=results_dir, format_strs=Config.baselines_log_format)
        rank = MPI.COMM_WORLD.Get_rank()

    # register the multi-agent env using the proper world and scenario settings
    # if args.continuous:
    #     register_env(Config.ma_c_env_id, Config.ma_c_env_entry_point, make_env_config(args))
    # else:
    #     register_env(Config.ma_env_id, Config.ma_env_entry_point, make_env_config(args))
    # register_env(args.env, 'gym_highway.multiagent_envs.multiagent:MultiAgentEnv', make_env("simple_spread", args))

    model, env = train(args, extra_args)
    env.close()
