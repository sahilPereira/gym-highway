"""
Helper functions for custom ppo2 model
"""

import tensorflow as tf

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
