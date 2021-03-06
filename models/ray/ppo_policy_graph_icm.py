from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import ray
from gym_highway.models.model import StateActionPredictor, StatePredictor
from gym_highway.envs.constants import constants
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.utils.explained_variance import explained_variance


class PPOLoss(object):
    def __init__(self,
                 action_space,
                 value_targets,
                 advantages,
                 actions,
                 logits,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 unsupType,
                 predictor,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        dist_cls, _ = ModelCatalog.get_action_dist(action_space, {})
        prev_dist = dist_cls(logits)
        # Make loss functions.
        logp_ratio = tf.exp(
            curr_action_dist.logp(actions) - prev_dist.logp(actions))
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = tf.reduce_mean(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = tf.reduce_mean(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = tf.reduce_mean(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(vf_loss)
            loss = tf.reduce_mean(-surrogate_loss + cur_kl_coeff * action_kl +
                                  vf_loss_coeff * vf_loss -
                                  entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = tf.reduce_mean(-surrogate_loss + cur_kl_coeff * action_kl -
                                  entropy_coeff * curr_entropy)
        self.loss = loss

        # computing predictor loss
        self.predloss = None
        if unsupType is not None:
            if 'state' in unsupType:
                self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
            else:
                self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                predictor.forwardloss * constants['FORWARD_LOSS_WT'])

class PPOPolicyGraphICM(LearningRateSchedule, TFPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 existing_inputs=None,
                 unsupType='action',
                 designHead='universe'):
        """
        Arguments:
            observation_space: Environment observation space specification.
            action_space: Environment action space specification.
            config (dict): Configuration values for PPO graph.
            existing_inputs (list): Optional list of tuples that specify the
                placeholders upon which the graph should be built upon.
        """
        self.unsup = unsupType is not None
        # self.cur_batch = None
        # self.cur_sample_batch = {}

        predictor = None
        numaction = action_space.n

        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = self.config["kl_coeff"]
        self.kl_target = self.config["kl_target"]

        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        if existing_inputs:
            obs_ph, value_targets_ph, adv_ph, act_ph, \
                logits_ph, vf_preds_ph, phi1, phi2, asample = existing_inputs[:9]

            # TODO: updates to account for s1, s2 and asample
            existing_state_in = existing_inputs[9:-1]
            existing_seq_lens = existing_inputs[-1]
        else:
            obs_ph = tf.placeholder(
                tf.float32,
                name="obs",
                shape=(None, ) + observation_space.shape)
            adv_ph = tf.placeholder(
                tf.float32, name="advantages", shape=(None, ))
            act_ph = ModelCatalog.get_action_placeholder(action_space)
            logits_ph = tf.placeholder(
                tf.float32, name="logits", shape=(None, logit_dim))
            vf_preds_ph = tf.placeholder(
                tf.float32, name="vf_preds", shape=(None, ))
            value_targets_ph = tf.placeholder(
                tf.float32, name="value_targets", shape=(None, ))
            phi1 = tf.placeholder(tf.float32, shape=(None, ) + observation_space.shape, name="phi1")
            phi2 = tf.placeholder(tf.float32, shape=(None, ) + observation_space.shape, name="phi2")
            asample = tf.placeholder(tf.float32, shape=(None, numaction), name="asample")

            existing_state_in = None
            existing_seq_lens = None
        self.observations = obs_ph


        if self.unsup:
            with tf.variable_scope("predictor"):
                if 'state' in unsupType:
                    self.local_ap_network = predictor = StatePredictor(phi1, phi2, asample, observation_space, numaction, designHead, unsupType)
                else:
                    self.local_ap_network = predictor = StateActionPredictor(phi1, phi2, asample, observation_space, numaction, designHead)

        self.model = pi = ModelCatalog.get_model(
            obs_ph,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens)


        # KL Coefficient
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

        self.logits = self.model.outputs
        curr_action_dist = dist_cls(self.logits)
        self.sampler = curr_action_dist.sample()

        if self.config["use_gae"]:
            if self.config["vf_share_layers"]:
                self.value_function = tf.reshape(
                    linear(self.model.last_layer, 1, "value",
                           normc_initializer(1.0)), [-1])
            else:
                vf_config = self.config["model"].copy()
                # Do not split the last layer of the value function into
                # mean parameters and standard deviation parameters and
                # do not make the standard deviations free variables.
                vf_config["free_log_std"] = False
                vf_config["use_lstm"] = False
                with tf.variable_scope("value_function"):
                    self.value_function = ModelCatalog.get_model(
                        obs_ph, 1, vf_config).outputs
                    self.value_function = tf.reshape(self.value_function, [-1])
        else:
            self.value_function = tf.zeros(shape=tf.shape(obs_ph)[:1])


        self.loss_obj = PPOLoss(
            action_space,
            value_targets_ph,
            adv_ph,
            act_ph,
            logits_ph,
            vf_preds_ph,
            curr_action_dist,
            self.value_function,
            self.kl_coeff,
            unsupType,
            predictor,
            entropy_coeff=self.config["entropy_coeff"],
            clip_param=self.config["clip_param"],
            vf_clip_param=self.config["vf_clip_param"],
            vf_loss_coeff=self.config["vf_loss_coeff"],
            use_gae=self.config["use_gae"])

        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])

        self.loss_in = [
            ("obs", obs_ph),
            ("value_targets", value_targets_ph),
            ("advantages", adv_ph),
            ("actions", act_ph),
            ("logits", logits_ph),
            ("vf_preds", vf_preds_ph),
            ("s1", phi1),
            ("s2", phi2),
            ("asample", asample),
        ]

        self.extra_inputs = ["s1","s2","asample"]

        # TODO: testing to see if this lets me pass inputs to ICM
        # self.variables = ray.experimental.TensorFlowVariables(self.loss_in, self.sess)

        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            loss=self.loss_obj.loss,
            loss_inputs=self.loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            seq_lens=self.model.seq_lens,
            max_seq_len=config["model"]["max_seq_len"])

        self.sess.run(tf.global_variables_initializer())
        self.explained_variance = explained_variance(value_targets_ph,
                                                     self.value_function)
        self.stats_fetches = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "total_loss": self.loss_obj.loss,
            "policy_loss": self.loss_obj.mean_policy_loss,
            "vf_loss": self.loss_obj.mean_vf_loss,
            "vf_explained_var": self.explained_variance,
            "kl": self.loss_obj.mean_kl,
            "entropy": self.loss_obj.mean_entropy
        }

    def copy(self, existing_inputs):
        """Creates a copy of self using existing input placeholders."""
        return PPOPolicyGraphICM(
            None,
            self.action_space,
            self.config,
            existing_inputs=existing_inputs)

    def extra_compute_action_fetches(self):
        return {"vf_preds": self.value_function, "logits": self.logits}

    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.sess)
        return self.kl_coeff_val

    def value(self, ob, *args):
        feed_dict = {self.observations: [ob], self.model.seq_lens: [1]}
        assert len(args) == len(self.model.state_in), \
            (args, self.model.state_in)
        for k, v in zip(self.model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.value_function, feed_dict)
        return vf[0]

    # def extra_compute_grad_feed_dict(self):
    #     feed_dict = {}
    #     # cur_batch = self.get_weights()

    #     # print("In extra_compute_grad_feed_dict: ", cur_batch.count)

    #     # if cur_batch:
    #     #     print("Current Batch shape: ", cur_batch.count)
    #     #     if self.unsup:
    #     # print("Current Batch obs: ", cur_batch["obs"][0])

    #     feed_dict[self.local_ap_network.s1] = cur_batch["obs"][:-1]
    #     feed_dict[self.local_ap_network.s2] = cur_batch["obs"][1:]

    #     one_hot_actions = np.eye(constants['NUM_ACTIONS'])[cur_batch["actions"]]
    #     feed_dict[self.local_ap_network.asample] = one_hot_actions[:-1]

    #     # return self.grad_feed_dict

    #     return feed_dict

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):

        # TODO: test to see whats in the sample_batch
        # for k,v in sample_batch.items():
        #     print(k,type(v))

        # collecting target for policy network
        rewards = np.asarray(sample_batch["rewards"])
        last_states = np.asarray(sample_batch["obs"])
        states = np.asarray(sample_batch["new_obs"])
        actions = np.asarray(sample_batch["actions"])
        bonuses = []


        one_hot_actions = np.eye(constants['NUM_ACTIONS'])[sample_batch["actions"]]
        one_hot_actions = np.array(one_hot_actions, dtype=np.float32)

        if self.local_ap_network is not None:
            for i in range(len(rewards)):
                bonus = self.local_ap_network.pred_bonus(self.sess, last_states[i], states[i], one_hot_actions[i])
                bonuses.append(bonus)
                # curr_tuple += [bonus, state]
                # life_bonus += bonus
                # ep_bonus += bonus

        if self.unsup:
            rewards += np.asarray(bonuses)

        sample_batch["rewards"] = rewards

        # if clip:
        #     rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])


        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = self.value(sample_batch["new_obs"][-1], *next_state)
        batch = compute_advantages(
            sample_batch,
            last_r,
            self.config["gamma"],
            self.config["lambda"],
            use_gae=self.config["use_gae"])
        return batch

    def gradients(self, optimizer):
        # return optimizer.compute_gradients(
        #     self._loss, colocate_gradients_with_ops=True)

        grads_and_vars = optimizer.compute_gradients(self.loss_obj.loss, 
            self.model.var_list, colocate_gradients_with_ops=True)

        predgrads = tf.gradients(self.loss_obj.predloss * 20.0, self.local_ap_network.var_list)

        # NOTE: not clipping policy grads as they were not clipped in original PPO graph
        # clip gradients
        if self.unsup:
            predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
            pred_grads_and_vars = list(zip(predgrads, self.local_ap_network.var_list))
            grads_and_vars = grads_and_vars + pred_grads_and_vars

        # return clipped_grads
        return grads_and_vars

    def get_initial_state(self):
        return self.model.state_init

    # def set_feed_dict(self, sample_batch):
    #     print("In set_feed_dict: ", sample_batch.count)
    #     print("is unsup?: ", self.unsup)

    #     # self.grad_feed_dict = {}
    #     if self.unsup:
    #         print("Current Batch obs: ", sample_batch["obs"][0])
    #         # feed_dict["obs"] = self.cur_batch["obs"]
    #         self.grad_feed_dict[self.local_ap_network.s1] = sample_batch["obs"][:-1]
    #         self.grad_feed_dict[self.local_ap_network.s2] = sample_batch["obs"][1:]

    #         # NOTE: the StateActionPredictor expects a one-hot encoding of the action space,
    #         # but my env is only providing a discrete action space.
    #         one_hot_actions = np.eye(constants['NUM_ACTIONS'])[sample_batch["actions"]]
    #         self.grad_feed_dict[self.local_ap_network.asample] = one_hot_actions[:-1]

        # self.grad_feed_dict = feed_dict

    def _get_loss_inputs_dict(self, batch):
        feed_dict = {}

        # Simple case
        extra_input_items = {}
        if not self.model.state_in:
            for k, ph in self.loss_in:
                if k in self.extra_inputs:
                    extra_input_items[k] = ph
                    continue
                feed_dict[ph] = batch[k]

            # if self.unsup:
            # print("Current Batch obs: ", batch["obs"][0])
            # print("Current Batch s1 len: ", len(batch["obs"][:-1]))
            # print("Current Batch s2 len: ", len(batch["obs"][1:]))
            # print("Current Batch type: ", type(batch["obs"]))
            
            feed_dict[extra_input_items["s1"]] = batch["obs"][:-1]
            feed_dict[extra_input_items["s2"]] = batch["obs"][1:]

            one_hot_actions = np.eye(constants['NUM_ACTIONS'])[batch["actions"]]
            # print("Current Batch asample len: ", len(one_hot_actions[:-1]))
            # print("Some one hot Actions: ", one_hot_actions[1:5])
            
            # one_hot_actions = np.float32(one_hot_actions)
            one_hot_actions = np.array(one_hot_actions, dtype=np.float32)

            # print("Some one hot Actions type: ", type(one_hot_actions))
            feed_dict[extra_input_items["asample"]] = one_hot_actions[:-1]

            # print("_get_loss_inputs_dict NON RNN feed_dict")
            return feed_dict

        # RNN case
        feature_keys = [k for k, v in self._loss_inputs]
        state_keys = [
            "state_in_{}".format(i) for i in range(len(self._state_inputs))
        ]
        feature_sequences, initial_states, seq_lens = chop_into_sequences(
            batch["eps_id"], [batch[k] for k in feature_keys],
            [batch[k] for k in state_keys], self._max_seq_len)
        for k, v in zip(feature_keys, feature_sequences):
            feed_dict[self._loss_input_dict[k]] = v
        for k, v in zip(state_keys, initial_states):
            feed_dict[self._loss_input_dict[k]] = v
        feed_dict[self._seq_lens] = seq_lens
        # print("_get_loss_inputs_dict RNN feed_dict")
        return feed_dict

    # def set_weights(self, weights):
    #     self.variables.set_weights(weights)

    # def get_weights(self):
    #     return self.variables.get_weights()