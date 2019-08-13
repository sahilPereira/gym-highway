"""
Modified DDPG learner for Multiple agents
ma_ddpg_learner built using ddpg_learner from baselines
"""

import functools
from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import baselines.common.tf_util as U
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.tf_util import (get_session, load_variables,
                                      save_variables)
from maddpg.common.distributions import make_pdtype

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class MADDPG(object):
    def __init__(self, name, actor, critic, memory, obs_space_n, act_space_n, agent_index, obs_rms, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):
        self.name = name
        self.num_agents = len(obs_space_n)
        self.agent_index = agent_index

        from gym import spaces
        continuous_ctrl = not isinstance(act_space_n[0], spaces.Discrete)
        # TODO: remove after testing
        assert continuous_ctrl

        # Multi-agent inputs
        self.actions = []
        self.obs0 = tf.placeholder(tf.float32, shape=(self.num_agents, None,) + obs_space_n[self.agent_index].shape, name="obs0")
        self.obs1 = tf.placeholder(tf.float32, shape=(self.num_agents, None,) + obs_space_n[self.agent_index].shape, name="obs1")

        # this is required to reshape obs and actions for concatenation
        obs_shape_list = [self.num_agents] + list(obs_space_n[self.agent_index].shape)
        if continuous_ctrl:
            act_shape_list = [self.num_agents] + list(act_space_n[self.agent_index].shape)
        else:
            act_shape_list = [self.num_agents] + [act_space_n[self.agent_index].n]
        self.obs_shape_prod = np.prod(obs_shape_list)
        self.act_shape_prod = np.prod(act_shape_list)

        for i in range(self.num_agents):
            if continuous_ctrl:
                self.actions.append(tf.placeholder(tf.float32, shape=[None] + list(act_space_n[i].shape), name="action"+str(i)))
            else:
                self.actions.append(make_pdtype(act_space_n[i]).sample_placeholder([None], name="action"+str(i)))

            # self.obs0.append(tf.placeholder(tf.float32, shape=(None,) + obs_space_n[i].shape, name="obs0_"+str(i)))
            # self.obs1.append(tf.placeholder(tf.float32, shape=(None,) + obs_space_n[i].shape, name="obs1_"+str(i)))

            # obs_shape_list.append(list(obs_space_n[i].shape))
        
        # we only provide single agent inputs for these placeholders
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')

        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        # TODO: need to update the replay buffer storage function to account for multiple agents
        if self.normalize_observations:
            self.obs_rms = obs_rms
        else:
            self.obs_rms = None
        
        # Need to transpose observations so we can normalize them
        # converts tensor to shape (batch_size, num_agents, space_size)
        # transose on dim 0 and 1, leave dim 2 unchanged
        obs0_t = tf.transpose(self.obs0, perm=[1, 0, 2])
        obs1_t = tf.transpose(self.obs1, perm=[1, 0, 2])
        actions_t = tf.transpose(self.actions, perm=[1, 0, 2])

        # each entry in obs_t is normalized wrt the agent
        normalized_obs0 = tf.clip_by_value(normalize(obs0_t, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(obs1_t, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        # convert the obs to original shape after normalization for convenience
        normalized_act_obs0 = tf.transpose(normalized_obs0, perm=[1, 0, 2])
        normalized_act_obs1 = tf.transpose(normalized_obs1, perm=[1, 0, 2])

        # need to specify exact shape, since we dont always pass batch size number of obs/act
        normalized_obs0_flat = tf.reshape(normalized_obs0, [-1, self.obs_shape_prod])
        normalized_obs1_flat = tf.reshape(normalized_obs1, [-1, self.obs_shape_prod])
        actions_t_flat = tf.reshape(actions_t, [-1, self.act_shape_prod])

        # Return normalization.
        # TODO: update this to handle multiple agents if required
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor_%d' % self.agent_index
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic_%d' % self.agent_index
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        # Each agents gets its own observation
        self.actor_tf = actor(normalized_act_obs0[self.agent_index])
        self.target_actor_tf = target_actor(normalized_act_obs1[self.agent_index])

        # Critic gets all observations
        self.normalized_critic_tf = critic(normalized_obs0_flat, actions_t_flat)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        # need to provide critic() with all actions
        act_input_n = self.actions + [] # copy actions
        act_input_n[self.agent_index] = self.actor_tf # update current agent action using its actor
        act_input_n_t = tf.transpose(act_input_n, perm=[1, 0, 2])
        act_input_n_t_flat = tf.reshape(act_input_n_t, [-1, self.act_shape_prod])
        self.normalized_critic_with_actor_tf = critic(normalized_obs0_flat, act_input_n_t_flat, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        # we need to use actions for all agents
        # target_act_input_n = self.actions + [] # copy actions
        # target_act_input_n[self.agent_index] = self.target_actor_tf # update current agent action using its target actor
        # target_act_input_n_t = tf.transpose(target_act_input_n, perm=[1, 0, 2])
        # target_act_input_n_t_flat = tf.reshape(target_act_input_n_t, [-1, self.act_shape_prod])

        # target actions are computed in train() and passed into action placeholder
        Q_obs1 = denormalize(target_critic(normalized_obs1_flat, actions_t_flat), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            # param noise is added to actor; hence obs for current agent is required
            self.setup_param_noise(normalized_act_obs0[self.agent_index])
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None # recurrent architectures not supported yet

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor_%d' % self.agent_index
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor_%d' % self.agent_index
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    # TODO: need to provide all observations to compute q
    def step(self, obs, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        # feed_dict={ph: [data] for ph, data in zip(self.obs0, obs)}
        # feed_dict = {self.obs0: [obs]}

        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])


        return action[0], q, None, None
    
    # TODO: test this
    # Computing this every time step may slow things
    def get_q_value(self, obs_n, act_n):
        # assuming computing q value for one state; hence need [] around data
        feed_dict={ph: [data] for ph, data in zip(self.obs0, obs_n)}
        act_dict={ph: [data] for ph, data in zip(self.actions, act_n)}
        feed_dict.update(act_dict)
        q = self.sess.run(self.critic_with_actor_tf, feed_dict=feed_dict)
        return q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        # reward *= self.reward_scale
        a_idx = self.agent_index
        self.memory.append(obs0[a_idx], action[a_idx], reward[a_idx], obs1[a_idx], terminal1[a_idx])
        # NOTE: calling update for each agent is ok, since the mean and std are uneffected
        # this is because the same obs are repeated num_agent times, which dont affect value
        if self.normalize_observations:
            # provide full obs for obs_rms update
            # obs0_shape = (len(obs0),)+obs0[a_idx].shape
            # assert obs0_shape == (self.num_agents,)+obs0[a_idx].shape
            self.obs_rms.update(np.array([obs0]))
    
    # TODO: not using this right now
    def update_obs_rms(self, obs0):
        if not self.normalize_observations:
            return
        B = obs0.shape[0]
        for b in range(B):
            # provide full obs for obs_rms update
            self.obs_rms.update(np.array([obs0[b]]))
        return

    def train(self, agents):
        # generate indices to access batches from all agents
        replay_sample_index = self.memory.generate_index(self.batch_size)

        # collect replay sample from all agents
        obs0_n = []
        obs1_n = []
        act_n = []
        target_act_n = []
        for i in range(self.num_agents):
            # Get a batch.
            batch = agents[i].memory.sample(batch_size=self.batch_size, index=replay_sample_index)
            obs0_n.append(batch['obs0'])
            obs1_n.append(batch['obs1'])
            act_n.append(batch['actions'])
        batch = self.memory.sample(batch_size=self.batch_size, index=replay_sample_index)

        # obs0_n_dict={ph: data for ph, data in zip(self.obs0, obs0_n)}
        # obs1_n_dict={ph: data for ph, data in zip(self.obs1, obs1_n)}
        # get target actions for each agent using obs1
        for i in range(self.num_agents):
            # target_obs1_n_dict={ph: data for ph, data in zip(agents[i].obs1, obs1_n)}
            target_acts = self.sess.run(agents[i].target_actor_tf, feed_dict={agents[i].obs1: obs1_n})
            # save the batch of target actions
            target_act_n.append(target_acts)
        
        # fill placeholders in obs1 with corresponding obs from each agent's replay buffer
        # self.obs1 and obs1_n are lists of size num_agents
        act_dict={ph: data for ph, data in zip(self.actions, act_n)}
        target_act_dict={ph: data for ph, data in zip(self.actions, target_act_n)}

        # feed dict for target_Q calculation
        feed_dict = {self.obs1: obs1_n}
        # feed_dict = obs1_n_dict
        feed_dict.update(target_act_dict)
        feed_dict.update({self.rewards: batch['rewards']})
        feed_dict.update({self.terminals1: batch['terminals1'].astype('float32')})

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict=feed_dict)
            # old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })

            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict=feed_dict)
            # target_Q = self.sess.run(self.target_Q, feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]

        # generate feed_dict for gradient and loss computation
        feed_dict = {self.obs0: obs0_n}
        # feed_dict = obs0_n_dict
        feed_dict.update(act_dict)
        feed_dict.update({self.critic_target: target_Q})

        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict=feed_dict)

        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess

    def agent_initialize(self, sess):
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)
        # setup saving and loading functions
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self, agents):
        if self.stats_sample is None:
            replay_sample_index = self.memory.generate_index(self.batch_size)
            # collect replay sample from all agents
            obs0_n, act_n = [], []
            for i in range(self.num_agents):
                batch = agents[i].memory.sample(batch_size=self.batch_size, index=replay_sample_index)
                obs0_n.append(batch['obs0'])
                act_n.append(batch['actions'])
            # generate feed_dict for multiple observations and actions
            # feed_dict={ph: data for ph, data in zip(self.obs0, obs0_n)}
            feed_dict = {self.obs0: obs0_n}

            actions_dict={ph: data for ph, data in zip(self.actions, act_n)}
            feed_dict.update(actions_dict)

            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = feed_dict
        values = self.sess.run(self.stats_ops, feed_dict=self.stats_sample)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self, agents):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        replay_sample_index = self.memory.generate_index(self.batch_size)
        obs0_n = []
        for i in range(self.num_agents):
            batch = agents[i].memory.sample(batch_size=self.batch_size, index=replay_sample_index)
            obs0_n.append(batch['obs0'])
        # feed_dict={ph: data for ph, data in zip(self.obs0, obs0_n)}
        feed_dict = {self.obs0: obs0_n}
        feed_dict.update({self.param_noise_stddev: self.param_noise.current_stddev})

        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict=feed_dict)

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
