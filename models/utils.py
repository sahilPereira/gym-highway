"""
General helper functions
"""

import datetime
import errno
import os
import os.path as osp
import random
import string
import json

import tensorflow as tf

import models.config as Config
from baselines.common.cmd_util import (common_arg_parser, make_env,
                                       make_vec_env, parse_unknown_args)


def activation_str_function(extra_args):
    '''
    Convert string activation into actual tf activation function
    '''
    activation_arg = extra_args['activation']
    if activation_arg == "relu":
        extra_args['activation'] = tf.nn.relu
    elif activation_arg == "tanh":
        extra_args['activation'] = tf.tanh
    elif activation_arg == "elu":
        extra_args['activation'] = tf.nn.elu
    else:
        extra_args['activation'] = tf.nn.sigmoid
    return extra_args

def id_generator(size=4, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def create_results_dir(args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    new_run_dir = "{}_{}_{}_{}".format(args.alg, args.env, current_time, id_generator())

    results_dir = "{}/{}".format(Config.results_dir, new_run_dir)
    results_dir = osp.expanduser(results_dir)
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return results_dir

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