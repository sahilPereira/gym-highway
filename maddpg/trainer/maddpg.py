import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        # Gradient computation mods
        # ---------------------------------------------------------------------------------------------
        obs_flat_shape = [len(obs_ph_n)*int(obs_ph_n[0].shape[-1])]
        act_flat_shape = [len(act_space_n)*int(act_space_n[0].shape[-1])]
        obs_flat_ph = tf.placeholder(tf.float32, shape=[None]+obs_flat_shape, name="obs_flat_input")
        act_flat_ph = tf.placeholder(tf.float32, shape=[None]+act_flat_shape, name="act_flat_input")

        q_vec_input = tf.concat([obs_flat_ph, act_flat_ph], axis=-1)
        serial_q = q_func(q_vec_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]

        # calculate gradient of serial q value wrt actions
        raw_grad = tf.gradients(serial_q, act_flat_ph)
        grad_norm = tf.divide(raw_grad, tf.norm(raw_grad))
        grad_norm_value = U.function([obs_flat_ph, act_flat_ph], grad_norm)
        # ---------------------------------------------------------------------------------------------

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act, 'grad_norm_value':grad_norm_value}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # get flattened obs and act shape
        act_shape = tf.shape(act_ph_n)
        act_serial = tf.concat(act_ph_n, 1)
        act_serial = tf.reshape(act_serial, [act_shape[1],act_shape[0]*act_shape[-1]])
        act_serial_values = U.function(act_ph_n, act_serial)
        
        obs_shape = tf.shape(obs_ph_n)
        obs_serial = tf.concat(obs_ph_n, 1)
        obs_serial = tf.reshape(obs_serial, [obs_shape[1],obs_shape[0]*obs_shape[-1]])
        obs_serial_values = U.function(obs_ph_n, obs_serial)

        obs_flat_shape = [len(obs_ph_n)*int(obs_ph_n[0].shape[-1])]
        act_flat_shape = [len(act_space_n)*int(act_space_n[0].shape[-1])]
        obs_flat_ph = tf.placeholder(tf.float32, shape=[None]+obs_flat_shape, name="obs_flat_input")
        act_flat_ph = tf.placeholder(tf.float32, shape=[None]+act_flat_shape, name="act_flat_input")

        target_input = tf.concat([obs_flat_ph, act_flat_ph], axis=-1)
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        # target_orig_q = q_func(q_input, 1, scope="target_orig_q_func", num_units=num_units)[:,0]
        target_q = q_func(target_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        # target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        target_q_values = U.function([obs_flat_ph, act_flat_ph], target_q)

        # calculate gradient of target q value wrt actions
        raw_grad = tf.gradients(target_q, act_flat_ph)
        grad_norm = tf.divide(raw_grad, tf.norm(raw_grad))
        grad_norm_value = U.function([obs_flat_ph, act_flat_ph], grad_norm)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values,'act_serial_values':act_serial_values, 
                                        'obs_serial_values':obs_serial_values, 'grad_norm_value':grad_norm_value}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, actor_lr=None, critic_lr=None, gamma=None, 
        num_units=None, rb_size=None, batch_size=None, max_episode_len=None, clip_norm=0.5, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        # training parameters
        self.actor_lr = actor_lr if actor_lr else args.lr
        self.critic_lr = critic_lr if critic_lr else args.lr
        self.gamma = gamma if gamma else args.gamma
        self.num_units = num_units if num_units else args.num_units
        self.rb_size = rb_size if rb_size else args.rb_size
        self.batch_size = batch_size if batch_size else args.batch_size
        self.max_episode_len = max_episode_len if max_episode_len else args.max_episode_len
        self.clip_norm = clip_norm

        # TODO: remove after testing
        import models.config as Config
        assert actor_lr == Config.maddpg_train_args['actor_lr']
        assert critic_lr == Config.maddpg_train_args['critic_lr']
        assert gamma == Config.maddpg_train_args['gamma']
        assert num_units == Config.maddpg_train_args['num_hidden']
        assert rb_size == Config.maddpg_train_args['rb_size']
        assert batch_size == Config.maddpg_train_args['batch_size']
        assert max_episode_len == Config.maddpg_train_args['nb_rollout_steps']
        assert clip_norm == Config.maddpg_train_args['clip_norm']
        
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.critic_lr),
            grad_norm_clipping=self.clip_norm,
            local_q_func=local_q_func,
            num_units=self.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.actor_lr),
            grad_norm_clipping=self.clip_norm,
            local_q_func=local_q_func,
            num_units=self.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(self.rb_size)
        self.max_replay_buffer_len = self.batch_size * self.max_episode_len
        self.replay_sample_index = None
        self.loss_names = ['q_loss', 'p_loss', 'mean_target_q', 'mean_rew', 'mean_target_q_next', 'std_target_q']

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        act_space = act.shape[-1]
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]

            # flatten multi agent actions and observations
            act_serial_vals = self.q_debug['act_serial_values'](*(target_act_next_n))
            obs_serial_vals = self.q_debug['obs_serial_values'](*(obs_next_n))
            assert len(act_serial_vals) == self.batch_size
            assert len(obs_serial_vals) == self.batch_size

            # compute L2 normalized partial derivatives of target Q function wrt actions
            # NOTE: this is done one sample at a time to prevent tf.gradient from summing over all target q values
            grad_norm_value = [self.q_debug['grad_norm_value'](*([[obs_serial_vals[j]]] + [[act_serial_vals[j]]])) for j in range(self.batch_size)]
            assert len(grad_norm_value) == self.batch_size
            
            # scale the raw gradients by alpha
            # TODO: set alpha during init or compute as function of policy or loss
            perturb = np.array(grad_norm_value) * 0.01
            
            # update leader actions using gradients
            for b in range(self.batch_size):
                # find all the leaders wrt current agent (agent_index)
                leading_agents = [[1.0]*act_space if obs_next_n[k][b][2] > obs_next_n[self.agent_index][b][2] else [0.0]*act_space for k in range(self.n)]
                # filter perturbations to only apply for leading agents
                # scale by L2 norm of original actions to prevent the perturb from overwhelming action
                epsilon = perturb[b].flatten() * np.array(leading_agents).flatten() * np.linalg.norm(act_serial_vals[b],2)
                act_serial_vals[b] += epsilon
            
            # target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q_next = self.q_debug['target_q_values'](*([obs_serial_vals] + [act_serial_vals]))
            target_q += rew + self.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # get current actions and observations flattened
        act_serial_vals = self.q_debug['act_serial_values'](*(act_n))
        obs_serial_vals = self.q_debug['obs_serial_values'](*(obs_n))
        # compute L2 normalized partial derivatives of Q function wrt actions
        grad_norm_value = [self.p_debug['grad_norm_value'](*([[obs_serial_vals[j]]] + [[act_serial_vals[j]]])) for j in range(self.batch_size)]
        assert len(grad_norm_value) == self.batch_size
        # scale the raw gradients by alpha
        perturb = np.array(grad_norm_value) * 0.01
        # update leader actions using these perturbations
        for b in range(self.batch_size):
            # find all the leaders wrt current agent (agent_index)
            leading_agents = [[1.0]*act_space if obs_next_n[k][b][2] > obs_next_n[self.agent_index][b][2] else [0.0]*act_space for k in range(self.n)]
            # filter perturbations to only apply for leading agents
            epsilon = perturb[b].flatten() * np.array(leading_agents).flatten() * np.linalg.norm(act_serial_vals[b],2)
            epsilon_n = [epsilon[k*act_space:(k*act_space)+act_space] for k in range(self.n)]
            # update each agent action for current batch sample "b"
            for k in range(self.n):
                act_n[k][b] += epsilon_n[k]

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
