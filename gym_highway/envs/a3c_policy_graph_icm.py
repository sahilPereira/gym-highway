'''
Copy from https://github.com/pathak22/noreward-rl 
Modifications to work with ray-project https://github.com/ray-project/ray 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gym_highway.envs.model import StateActionPredictor, StatePredictor
from gym_highway.envs.constants import constants

# imports from ray a3c_tf_policy_graph

"""Note: Keep in sync with changes to VTracePolicyGraph."""
import gym

import ray
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.models.catalog import ModelCatalog

class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 unsupType,
                 predictor,
                 vf_loss_coeff=0.5,
                 entropy_coeff=-0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff +
                           self.entropy * entropy_coeff)

        # computing predictor loss
        self.predloss = None
        if unsupType is not None:
            if 'state' in unsupType:
                self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
            else:
                self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                predictor.forwardloss * constants['FORWARD_LOSS_WT'])

class A3CPolicyGraphICM(LearningRateSchedule, TFPolicyGraph):
    def __init__(self, observation_space, action_space, config, unsupType='action', envWrap=False, designHead='universe', noReward=False):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.unsup = unsupType is not None
        self.cur_batch = None

        predictor = None
        numaction = action_space.n

        config = dict(ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG, **config)
        self.config = config
        self.sess = tf.get_default_session()

        # Setup the policy
        # =====================================================================
        self.observations = tf.placeholder(tf.float32, [None] + list(observation_space.shape))
        dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, self.config["model"])

        # NOTE: value function and trainable variables are defined in self.model
        # Define the policy network
        self.model = pi = ModelCatalog.get_model(self.observations, logit_dim, self.config["model"])
        action_dist = dist_class(self.model.outputs)

        # Define S/S+A predictor network
        # TODO: update the predictors to work with non-image data
        # TODO UPDATE: modified the predictors to work with vectorized data
        if self.unsup:
            with tf.variable_scope("predictor"):
                if 'state' in unsupType:
                    self.local_ap_network = predictor = StatePredictor(observation_space.shape, numaction, designHead, unsupType)
                else:
                    self.local_ap_network = predictor = StateActionPredictor(observation_space.shape, numaction, designHead)

        # Setup the policy loss
        # =====================================================================
        if isinstance(action_space, gym.spaces.Box):
            ac_size = action_space.shape[0]
            actions = tf.placeholder(tf.float32, [None, ac_size], name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            actions = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise UnsupportedSpaceException(
                "Action space {} is not supported for A3C.".format(
                    action_space))
        advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.v_target = tf.placeholder(tf.float32, [None], name="v_target")

        # compute policy loss and predictor loss
        self.loss = A3CLoss(action_dist, actions, advantages, self.v_target,
                            self.model.vf, unsupType, predictor, self.config["vf_loss_coeff"],
                            self.config["entropy_coeff"])

        # Initialize TFPolicyGraph
        loss_in = [
            ("obs", self.observations),
            ("actions", actions),
            ("advantages", advantages),
            ("value_targets", self.v_target),
        ]
        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.observations,
            action_sampler=action_dist.sample(),
            loss=self.loss.total_loss,
            loss_inputs=loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            seq_lens=self.model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.stats_fetches = {
            "stats": {
                "cur_lr": tf.cast(self.cur_lr, tf.float64),
                "policy_loss": self.loss.pi_loss,
                "policy_entropy": self.loss.entropy,
                "grad_gnorm": tf.global_norm(self._grads),
                "var_gnorm": tf.global_norm(self.model.var_list),
                "vf_loss": self.loss.vf_loss,
                "vf_explained_var": explained_variance(self.v_target, self.model.vf),
            },
        }

        self.sess.run(tf.global_variables_initializer())

    def extra_compute_action_fetches(self):
        return {"vf_preds": self.model.vf}

    def value(self, ob, *args):
        feed_dict = {self.observations: [ob], self.model.seq_lens: [1]}
        assert len(args) == len(self.model.state_in), \
            (args, self.model.state_in)
        for k, v in zip(self.model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.model.vf, feed_dict)
        return vf[0]

    def gradients(self, optimizer):
        # compute gradients
        grads = tf.gradients(self.loss.total_loss, self.model.var_list)
        predgrads = tf.gradients(self.loss.predloss * 20.0, self.local_ap_network.var_list)

        # clip gradients
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        # clipped_grads = list(zip(self.grads, self.var_list))
        grads_and_vars = list(zip(self.grads, self.model.var_list))
        if self.unsup:
            predgrads, _ = tf.clip_by_global_norm(predgrads, self.config["grad_clip"])
            pred_grads_and_vars = list(zip(predgrads, self.local_ap_network.var_list))
            grads_and_vars = grads_and_vars + pred_grads_and_vars

        # return clipped_grads
        return grads_and_vars

    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    # extra inputs for the ICM network
    def extra_compute_grad_feed_dict(self):
        feed_dict = {}
        if self.cur_batch:
            if self.unsup:
                # feed_dict["obs"] = self.cur_batch["obs"]
                feed_dict[self.local_ap_network.s1] = self.cur_batch["obs"][:-1]
                feed_dict[self.local_ap_network.s2] = self.cur_batch["obs"][1:]

                # NOTE: the StateActionPredictor expects a one-hot encoding of the action space,
                # but my env is only providing a discrete action space.
                # one_hot_actions = tf.one_hot(self.cur_batch["actions"], constants['NUM_ACTIONS'])
                one_hot_actions = np.eye(constants['NUM_ACTIONS'])[self.cur_batch["actions"]]
                feed_dict[self.local_ap_network.asample] = one_hot_actions[:-1]
        return feed_dict

    def get_initial_state(self):
        return self.model.state_init

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        self.cur_batch = sample_batch
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = self.value(sample_batch["new_obs"][-1], *next_state)
        return compute_advantages(sample_batch, last_r, self.config["gamma"],
                                  self.config["lambda"])
