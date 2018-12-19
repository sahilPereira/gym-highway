from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import ray
from gym_highway.envs.model import StateActionPredictor, StatePredictor
from gym_highway.envs.constants import constants
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.utils.explained_variance import explained_variance


class ICMLoss(object):
    def __init__(self,
                 action_space,
                 unsupType,
                 predictor):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
        """
        # computing predictor loss
        self.predloss = None
        if unsupType is not None:
            if 'state' in unsupType:
                self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
            else:
                self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                predictor.forwardloss * constants['FORWARD_LOSS_WT'])

class ICMGraph(LearningRateSchedule, TFPolicyGraph):
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
        predictor = None
        numaction = action_space.n

        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config

        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        obs_ph = tf.placeholder(tf.float32, name="obs", shape=(None, ) + observation_space.shape)
        act_ph = ModelCatalog.get_action_placeholder(action_space)

        # print("Observations space:", observation_space)
        print("Observations shape:", tf.shape(self.observations)[0]) # tf.shape(x)[0]
        print("Observations shape:", self.observations.shape[1:])
        print("obs_ph shape:", tf.shape(obs_ph)[:1])
        with tf.variable_scope("predictor"):
            if 'state' in unsupType:
                self.local_ap_network = predictor = StatePredictor(tf.shape(self.observations)[0], numaction, designHead, unsupType)
            else:
                self.local_ap_network = predictor = StateActionPredictor(tf.shape(self.observations)[0], numaction, designHead)


        self.logits = self.local_ap_network.outputs
        curr_action_dist = dist_cls(self.logits)
        self.sampler = curr_action_dist.sample()


        self.loss_obj = ICMLoss(action_space, unsupType, predictor)

        LearningRateSchedule.__init__(self, self.config["lr"], self.config["lr_schedule"])

        self.loss_in = [
            ("obs", obs_ph),
            ("actions", act_ph),
            ("phi1", self.local_ap_network.s1),
            ("phi2", self.local_ap_network.s2),
            ("asample", self.local_ap_network.asample),
        ]

        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            loss=self.loss_obj.predloss,
            loss_inputs=self.loss_in,
            max_seq_len=config["model"]["max_seq_len"])

        self.sess.run(tf.global_variables_initializer())

        self.stats_fetches = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "total_loss": self.loss_obj.predloss
            # "policy_loss": self.loss_obj.mean_policy_loss,
            # "vf_loss": self.loss_obj.mean_vf_loss,
            # "vf_explained_var": self.explained_variance,
            # "kl": self.loss_obj.mean_kl,
            # "entropy": self.loss_obj.mean_entropy
        }

    def copy(self, existing_inputs):
        """Creates a copy of self using existing input placeholders."""
        return ICMGraph(
            None,
            self.action_space,
            self.config,
            existing_inputs=existing_inputs)

    # def extra_compute_action_fetches(self):
    #     return {"vf_preds": self.value_function, "logits": self.logits}

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

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
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

    def _get_loss_inputs_dict(self, batch):
        feed_dict = {}

        # Simple case
        if not self.model.state_in:
            for k, ph in self.loss_in:
                feed_dict[ph] = batch[k]

            # if self.unsup:
            print("Current Batch obs: ", batch["obs"][0])
            print("Current Batch s1 len: ", len(batch["obs"][:-1]))
            print("Current Batch s2 len: ", len(batch["obs"][1:]))
            print("Current Batch type: ", type(batch["obs"]))
            
            feed_dict[self.local_ap_network.s1] = batch["obs"][:-1]
            feed_dict[self.local_ap_network.s2] = batch["obs"][1:]

            one_hot_actions = np.eye(constants['NUM_ACTIONS'])[batch["actions"]]
            print("Current Batch asample len: ", len(one_hot_actions[:-1]))
            print("Some one hot Actions: ", one_hot_actions[1:5])
            
            one_hot_actions = np.float32(one_hot_actions)

            print("Some one hot Actions type: ", type(one_hot_actions))
            feed_dict[self.local_ap_network.asample] = list(one_hot_actions[:-1])

            print("_get_loss_inputs_dict NON RNN feed_dict")
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
        print("_get_loss_inputs_dict RNN feed_dict")
        return feed_dict
