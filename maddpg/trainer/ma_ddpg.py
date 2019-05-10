"""
Modified DDPG for Multiple agents
MADDPG built using DDPG from baselines
"""

import os
import os.path as osp
import time
from collections import deque
import pickle

from maddpg.trainer.ma_ddpg_learner import MADDPG
# from baselines.ddpg.models import Actor, Critic
# from baselines.ddpg.memory import Memory
from maddpg.trainer.ma_models import Actor, Critic
from maddpg.trainer.ma_memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from models import config as Config
from gym import spaces
import tensorflow as tf
from baselines.common.mpi_running_mean_std import RunningMeanStd

from baselines import logger
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def get_noise(noise_type, nb_actions):
    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return action_noise, param_noise

def learn(network, env,
          seed=None,
          total_timesteps=None,
          num_agents=1,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          save_interval=100,
          load_path=None,
          num_adversaries=0,
          adv_policy='maddpg',
          good_policy='maddpg',
          **network_kwargs):

    set_global_seeds(seed)

    continuous_ctrl = not isinstance(env.action_space[0], spaces.Discrete)
    assert continuous_ctrl
    
    nb_actions = env.action_space[0].shape[-1] if continuous_ctrl else env.action_space[0].n
    
    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    # NOTE: This is mainly for debugging
    obs_shape_n = [env.observation_space[i].shape for i in range(num_agents)]
    print("Num of observations: {}".format(len(obs_shape_n)))
    print('Observation shapes {}'.format(obs_shape_n))

    sess = U.get_session()
    # create one obs_rms used by all agents
    obs_shape = (num_agents,)+env.observation_space[0].shape
    with tf.variable_scope('obs_rms'):
        obs_rms = RunningMeanStd(shape=obs_shape)
    trainers = []
    for i in range(num_agents):
        # get action shape for an agent
        action_shape = env.action_space[i].shape if continuous_ctrl else (nb_actions, )

        # TODO: will have to modify the critic and actor to work with batches for multiple agents
        memory = Memory(limit=int(1e6), action_shape=action_shape, observation_shape=env.observation_space[i].shape)
        critic = Critic(name="critic_%d" % i, network=network, **network_kwargs)
        actor = Actor(nb_actions, name="actor_%d" % i, network=network, **network_kwargs)

        # get action and parameter noise type
        action_noise, param_noise = get_noise(noise_type, nb_actions)

        # TODO: remove after testing
        assert param_noise is not None

        # TODO: need to update the placeholders in MADDPG based off of ddpg_learner
        # replay buffer, actor and critic are defined for each agent in trainers
        agent = MADDPG("agent_%d" % i, actor, critic, memory, env.observation_space, env.action_space, i, obs_rms,
            gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
            batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
            actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
            reward_scale=reward_scale)
        
        # Prepare agent
        agent.initialize(sess)
        trainers.append(agent)
    
    # TODO: test if this actually works
    # Test by running the trained models (num_agents>=2)
    sess.run(tf.global_variables_initializer())
    for agent in trainers:
        agent.agent_initialize(sess)

    # max_action = env.action_space.high
    max_action = 1
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    logger.info('Using agents with the following configuration:')
    logger.info(str(trainers[0].__dict__.items()))

    if load_path is not None:
        load_path = "{}/{}/checkpoints/checkpoints-final".format(Config.results_dir, load_path)
        load_path = osp.expanduser(load_path)
        # TODO: test if this works
        # assuming this will work cause save and load are done at the global level and any agent 
        # should be able to load the full state
        trainers[0].load(load_path)
    
    sess.graph.finalize()

    # reset all agents
    for agent in trainers:
        agent.reset()

    obs_n = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs_n.shape[0]

    # initialize metric tracking parameters
    episode_reward = np.zeros((nenvs, len(trainers)), dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar
    epoch = 0

    start_time = time.time()
    epoch_episode_rewards = np.zeros(len(trainers), dtype = np.float32)
    epoch_episode_steps = np.zeros(nenvs, dtype = int)
    # TODO: update epoch_actions to find std between actions over time
    epoch_actions = 0.0
    epoch_qs = np.zeros(len(trainers), dtype = np.float32)
    epoch_episodes = 0

    episode_rewards_history = deque(maxlen=100)
    episode_steps_history = deque(maxlen=100)
    episode_agent_rewards_history = [deque(maxlen=100) for _ in range(len(trainers))]

    print('Starting iterations...')
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                for agent in trainers:
                    agent.reset()
            for t_rollout in range(nb_rollout_steps):
                actions_n = []
                # q_n = np.zeros(len(trainers), dtype = np.float32)
                for i in range(nenvs):
                    # create n copies of full obs where n = num agents; memory is not an issue for this simulation
                    rep_obs = np.stack([obs_n[i] for _ in range(len(trainers))])
                    # Predict next actions and q vals for all agents in current env
                    # call step() with each agent and full observation; only get action "[0]" from this call
                    action_q_list = [agent.step(obs, apply_noise=True, compute_Q=False)[0] for agent, obs in zip(trainers, rep_obs)]
                    # store actions and q vals in respective lists
                    actions_n.append(action_q_list)
                    # we care about the overall q value for each agent
                    # print("action_q_list: ", action_q_list)
                    # print("q_n: ", np.array(action_q_list)[:,1])
                    # q_n += np.array(action_q_list)[:,1]
                
                # confirm actions_n is nenvs x num_agents x len(Action)
                # print("actions_n: ", actions_n)
                # print("actions_n: ", actions_n[0][1])
                # print((len(actions_n),len(actions_n[0]),len(actions_n[0][0])))
                # print((nenvs, num_agents, nb_actions))
                assert (len(actions_n),len(actions_n[0]),len(actions_n[0][0])) == (nenvs, num_agents, nb_actions)

                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(actions_n)

                # sum of rewards for each env
                episode_reward += [r for r in rew_n]
                episode_step += 1
                # epoch_qs += [q / float(nenvs) for q in q_n]

                # Book-keeping
                for i, agent in enumerate(trainers):
                    # for b in range(nenvs):
                    #     # save experience from all envs for each agent
                    #     agent.store_transition(obs_n[b][i], actions_n[b][i], rew_n[b][i], new_obs_n[b][i], done_n[b][i], None)
                    agent.store_transition(obs_n, actions_n, rew_n, new_obs_n, done_n)
                obs_n = new_obs_n

                # looping over nenvs
                for d in range(len(done_n)):
                    if any(done_n[d]):
                        # Episode done.
                        epoch_episode_rewards += episode_reward[d]
                        # keep track of individual agent reward history
                        for i in range(num_agents):
                            episode_agent_rewards_history[i].append(episode_reward[d][i])
                        # track combined reward history of all agents
                        episode_rewards_history.append(sum(episode_reward[d]))
                        # episode steps over all runs
                        epoch_episode_steps[d] += episode_step[d]
                        # episode steps for last 100 runs
                        episode_steps_history.append(episode_step[d])
                        # reset env specific list for next run
                        episode_reward[d] = np.zeros(len(trainers), dtype = np.float32)
                        episode_step[d] = 0
                        # increment counters
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            for agent in trainers:
                                agent.reset()
                
                # update timestep
                t += 1

            # Train.
            epoch_actor_losses = [[] for _ in range(len(trainers))]
            epoch_critic_losses = [[] for _ in range(len(trainers))]
            epoch_adaptive_distances = [[] for _ in range(len(trainers))]

            for t_train in range(nb_train_steps):
                for i, agent in enumerate(trainers):
                    # Adapt param noise, if necessary.
                    if agent.memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise(trainers)
                        epoch_adaptive_distances[i].append(distance)

                    cl, al = agent.train(trainers)
                    epoch_critic_losses[i].append(cl)
                    epoch_actor_losses[i].append(al)
                    agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_action_step = eval_action if continuous_ctrl else np.argmax(max_action * eval_action, axis=1)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(eval_action_step)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        combined_stats = {}
        # get stats for all agents
        for i, agent in enumerate(trainers):
            stats = agent.get_stats(trainers)
            for k,v in stats.items():
                combined_stats["{}st/ag{}_{}".format(Config.tensorboard_rootdir,i,k)] = v

            # agent specific rollout metrics
            combined_stats["{}ro/ag{}_return".format(Config.tensorboard_rootdir,i)] = epoch_episode_rewards[i] / float(episodes)
            combined_stats["{}ro/ag{}_return_history".format(Config.tensorboard_rootdir,i)] = np.mean(episode_agent_rewards_history[i])
            # combined_stats["{}ro/ag{}_Q_mean".format(Config.tensorboard_rootdir,i)] = epoch_qs[i] / float(t)

            # agent specific training metrics
            combined_stats["{}tr/ag{}_loss_actor".format(Config.tensorboard_rootdir,i)] = np.mean(epoch_actor_losses[i])
            combined_stats["{}tr/ag{}_loss_critic".format(Config.tensorboard_rootdir,i)] = np.mean(epoch_critic_losses[i])
            combined_stats["{}tr/ag{}_param_noise_distance".format(Config.tensorboard_rootdir,i)] = np.mean(epoch_adaptive_distances[i])
        
        combined_stats[Config.tensorboard_rootdir+'ro/return'] = np.mean(epoch_episode_rewards) / float(episodes)
        combined_stats[Config.tensorboard_rootdir+'ro/return_history'] = np.mean(episode_rewards_history)
        combined_stats[Config.tensorboard_rootdir+'ro/episode_steps'] = np.mean(epoch_episode_steps) / float(episodes)
        combined_stats[Config.tensorboard_rootdir+'ro/episode_steps_history'] = np.mean(episode_steps_history)
        # combined_stats[Config.tensorboard_rootdir+'ro/Q_mean'] = np.mean(epoch_qs) / float(t)
        combined_stats[Config.tensorboard_rootdir+'tr/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats[Config.tensorboard_rootdir+'tr/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats[Config.tensorboard_rootdir+'tr/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats[Config.tensorboard_rootdir+'to/duration'] = duration
        combined_stats[Config.tensorboard_rootdir+'to/steps_per_second'] = float(t) / float(duration)
        combined_stats[Config.tensorboard_rootdir+'to/episodes'] = episodes
        combined_stats[Config.tensorboard_rootdir+'ro/episodes'] = epoch_episodes
        combined_stats[Config.tensorboard_rootdir+'ro/std_return'] = np.std(epoch_episode_rewards)
        # combined_stats[Config.tensorboard_rootdir+'ro/actions_mean'] = epoch_actions / float(t)
        # combined_stats[Config.tensorboard_rootdir+'rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats[Config.tensorboard_rootdir+'eval/return'] = eval_episode_rewards
            combined_stats[Config.tensorboard_rootdir+'eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats[Config.tensorboard_rootdir+'eval/Q'] = eval_qs
            combined_stats[Config.tensorboard_rootdir+'eval/episodes'] = len(eval_episode_rewards)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats[Config.tensorboard_rootdir+'to/epochs'] = epoch + 1
        combined_stats[Config.tensorboard_rootdir+'to/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        
        if save_interval and (epoch % save_interval == 0) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%epoch)
            print('Saving to', savepath)
            # TODO: test if this actually saves all the agents
            # i assume it does because save() has access to global variables
            trainers[0].save(savepath)

    return trainers
