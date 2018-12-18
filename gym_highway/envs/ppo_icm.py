from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents import Agent
from gym_highway.envs.ppo_policy_graph_icm import PPOPolicyGraphICM
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.utils import merge_dicts
from ray.rllib.optimizers import SyncSamplesOptimizer, LocalMultiGPUOptimizer
from ray.tune.trial import Resources

# class PPOAgentICM(PPOAgent):
#     """Multi-GPU optimized implementation of PPO in TensorFlow."""

#     _agent_name = "PPO_ICM"
#     _default_config = DEFAULT_CONFIG
#     _policy_graph = PPOPolicyGraphICM

class PPOAgentICM(Agent):
    """Multi-GPU optimized implementation of PPO in TensorFlow."""

    # _agent_name = "PPO"
    # _default_config = DEFAULT_CONFIG
    # _policy_graph = PPOPolicyGraph

    _agent_name = "PPO_ICM"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PPOPolicyGraphICM

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(
            cpu=1,
            gpu=cf["num_gpus"],
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"],
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"])

    def _init(self):
        self._validate_config()
        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, self._policy_graph)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, self._policy_graph, self.config["num_workers"], {
                "num_cpus": self.config["num_cpus_per_worker"],
                "num_gpus": self.config["num_gpus_per_worker"]
            })
        if self.config["simple_optimizer"]:
            self.optimizer = SyncSamplesOptimizer(
                self.local_evaluator, self.remote_evaluators, {
                    "num_sgd_iter": self.config["num_sgd_iter"],
                    "train_batch_size": self.config["train_batch_size"]
                })
        else:
            self.optimizer = LocalMultiGPUOptimizer(
                self.local_evaluator, self.remote_evaluators, {
                    "sgd_batch_size": self.config["sgd_minibatch_size"],
                    "num_sgd_iter": self.config["num_sgd_iter"],
                    "num_gpus": self.config["num_gpus"],
                    "train_batch_size": self.config["train_batch_size"],
                    "standardize_fields": ["advantages"],
                })

    def _validate_config(self):
        waste_ratio = (
            self.config["sample_batch_size"] * self.config["num_workers"] /
            self.config["train_batch_size"])
        if waste_ratio > 1:
            msg = ("sample_batch_size * num_workers >> train_batch_size. "
                   "This means that many steps will be discarded. Consider "
                   "reducing sample_batch_size, or increase train_batch_size.")
            if waste_ratio > 1.5:
                raise ValueError(msg)
            else:
                print("Warning: " + msg)
        if self.config["sgd_minibatch_size"] > self.config["train_batch_size"]:
            raise ValueError(
                "Minibatch size {} must be <= train batch size {}.".format(
                    self.config["sgd_minibatch_size"],
                    self.config["train_batch_size"]))
        if (self.config["batch_mode"] == "truncate_episodes"
                and not self.config["use_gae"]):
            raise ValueError(
                "Episode truncation is not supported without a value function")

    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        fetches = self.optimizer.step()
        if "kl" in fetches:
            # single-agent
            self.local_evaluator.for_policy(
                lambda pi: pi.update_kl(fetches["kl"]))
        else:
            # multi-agent
            self.local_evaluator.foreach_trainable_policy(
                lambda pi, pi_id: pi.update_kl(fetches[pi_id]["kl"]))

        # samples = self.local_evaluator.sample()

        res = self.optimizer.collect_metrics()
        res.update(
            timesteps_this_iter=self.optimizer.num_steps_sampled - prev_steps,
            info=dict(fetches, **res.get("info", {})))
        return res
