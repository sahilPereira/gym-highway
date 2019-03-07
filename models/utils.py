"""
General helper functions
"""

import datetime
import errno
import os
import os.path as osp
import random
import string

import tensorflow as tf

import models.config as Config


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

def id_generator(size=10, chars=string.ascii_lowercase + string.digits):
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
