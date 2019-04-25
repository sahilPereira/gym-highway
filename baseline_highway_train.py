import sys
import os
import os.path as osp
import multiprocessing
import json
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from importlib import import_module
import datetime
import random, string
import errno

# imports from baselines.run
import gym
from gym.envs.registration import register
import gym_highway
from gym_highway.envs import HighwayEnv

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from baselines.common.vec_env.vec_normalize import VecNormalize

import tensorflow as tf
import models.config as Config
from models.utils import activation_str_function, create_results_dir, id_generator

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_game_envs = defaultdict(set)


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    # if extra_args["use_icm"]:
    #     # if using ICM use ppo2_icm submodule in ppo2
    #     learn = get_alg_module(args.alg, Config.icm_submodule).learn
    # else:
    learn = get_learn_function(args.alg)
    
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    # TODO: removed since we dont save any video intervals
    # if args.save_video_interval != 0:
    #     env = VecVideoRecorder(env, osp.join(logger.Logger.CURRENT.dir, "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    return model, env

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

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

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def register_env(c_id, c_entry_point, c_kwargs):
    '''
    Register a gym environment with new id, entry point and kwargs
    '''
    register( id=c_id, entry_point=c_entry_point, kwargs=c_kwargs)

def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def save_configs(results_dir, args, extra_args):
    config_save_path = "{}/configs.txt".format(results_dir)

    args_copy = {k: v for k,v in vars(args).items()}
    extra_args_copy = {k: v for k,v in extra_args.items()}
    extra_args_copy['activation'] = str(extra_args_copy['activation'])
    with open(config_save_path, mode='w', encoding='utf-8') as f:
        json.dump([args_copy, extra_args_copy], f, indent=4)
    return True

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def custom_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default=Config.env_cont_id)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6)
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel', default=Config.num_workers, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Location to save trained model', default=None, type=str)
    parser.add_argument('--save_model', default=True, action='store_false')
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--extra_import', help='Extra module to import to access external environments', type=str, default=None)
    parser.add_argument('--desc', help='Description of experiment', type=str, default=None)
    return parser

def main(args):

    arg_parser = custom_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    
    # add custom training arguments for ppo2 algorithm
    # extra_args = Config.ppo2_train_args
    extra_args = Config.ddpg_train_args
    extra_args = activation_str_function(extra_args)

    # update extra_args with command line argument overrides
    extra_args.update(parse_cmdline_kwargs(unknown_args))

    # create a separate result dir for each run
    results_dir = create_results_dir(args)

    # save configurations
    save_configs(results_dir, args, extra_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        # logger.configure()
        logger.configure(dir=results_dir, format_strs=Config.baselines_log_format)
    else:
        # logger.configure(format_strs=[])
        logger.configure(dir=results_dir, format_strs=Config.baselines_log_format)
        rank = MPI.COMM_WORLD.Get_rank()

    if args.play:
        register_env(Config.env_cont_id, Config.env_cont_entry_point, Config.env_play_kwargs)
    else:
        register_env(Config.env_cont_id, Config.env_cont_entry_point, Config.env_train_kwargs)

    model, env = train(args, extra_args)
    env.close()

    if args.save_model and rank == 0 and not args.test:
        save_path = "{}/checkpoints/checkpoints-final".format(results_dir)
        # save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.test:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                # actions, _, _, _ = model.step(obs)
                # for DDPG
                actions, _, _, _ = model.step(obs, apply_noise=False, compute_Q=True)

            obs, _, done, _ = env.step(actions)
            
            # Not required since the gym highway environment renders based on init param
            # env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)