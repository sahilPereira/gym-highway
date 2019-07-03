import numpy as np
from multiprocessing import Process, Pipe
from . import VecTrainer, CloudpickleWrapper

def worker(remote, parent_remote, agent_fn_wrapper):
    parent_remote.close()
    agent = agent_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            # new functions
            if cmd == 'step':
                obs, apply_noise, compute_Q = data
                action, q = agent.step(obs, apply_noise=apply_noise, compute_Q=compute_Q)
                remote.send((action, q))
            elif cmd == 'generate_index':
                replay_sample_index = agent.generate_index()
                remote.send(replay_sample_index)
            elif cmd == 'sample_batch':
                batch = agent.sample_batch(data)
                remote.send(batch)
            elif cmd == 'store_transition':
                obs_n, actions_n, rew_n, new_obs_n, done_n = data
                ret = agent.store_transition(obs_n, actions_n, rew_n, new_obs_n, done_n)
                remote.send(ret)
            elif cmd == 'initialize':
                ret = agent.initialize(data)
                remote.send(ret)
            elif cmd == 'agent_initialize':
                ret = agent.agent_initialize(data)
                remote.send(ret)
            elif cmd == 'adapt_param_noise':
                mean_distance = agent.adapt_param_noise(data)
                remote.send(mean_distance)
            elif cmd == 'train':
                # TODO: update with correct parameters
                critic_loss, actor_loss = agent.train(data)
                remote.send((critic_loss, actor_loss))
            elif cmd == 'get_stats':
                # TODO: update with correct parameters
                stats = agent.get_stats(data)
                remote.send(stats)
            elif cmd == 'save':
                # TODO: update with correct parameters (savepath)
                res = agent.save(data)
                remote.send(res)
            elif cmd == 'load':
                # TODO: update with correct parameters (loadpath)
                res = agent.load(data)
                remote.send(res)
            elif cmd == 'reset':
                ret = agent.reset()
                remote.send(ret)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecMADDPG worker: got KeyboardInterrupt')
    finally:
        print('SubprocVecMADDPG worker has closed')
        # env.close()


class SubprocVecMADDPG(VecTrainer):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, train_fns, spaces=None):
        """
        Arguments:

        train_fns: iterable of callables -  functions that create trainers to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        ntrainers = len(train_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(ntrainers)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(train_fn)))
                   for (work_remote, remote, train_fn) in zip(self.work_remotes, self.remotes, train_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # self.remotes[0].send(('get_spaces', None))
        # these are lists of observations and actions
        # observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        # self.specs = [f().spec for f in train_fns]
        VecTrainer.__init__(self, len(train_fns))

    def step_async(self, obs_n, apply_noise, compute_Q):
        self._assert_not_closed()
        # since we are working with multiple agents, each "obs_n"
        # is a list of observations for each agent in that env
        for remote, obs in zip(self.remotes, obs_n):
            remote.send(('step', [obs, apply_noise, compute_Q]))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def generate_index_async(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('generate_index', None))
        self.waiting = True

    def generate_index_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # obs, rews, dones, infos = zip(*results)
        return results

    def sample_batch_async(self, replay_sample_index):
        self._assert_not_closed()
        for remote, sample_index in zip(self.remotes, replay_sample_index):
            remote.send(('sample_batch', sample_index))
        self.waiting = True

    def sample_batch_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # batch = zip(*results)
        return np.stack(results)

    def initialize_async(self, sess):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('initialize', sess))
        self.waiting = True

    def initialize_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return

    def agent_initialize_async(self, sess):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('agent_initialize', sess))
        self.waiting = True

    def agent_initialize_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return

    def store_transition_async(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
        self._assert_not_closed()
        for remote, obs, actions, rew, new_obs, done in zip(self.remotes, obs_n, actions_n, rew_n, new_obs_n, done_n):
            remote.send(('store_transition', (obs, actions, rew, new_obs, done)))
        self.waiting = True

    def store_transition_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return

    def adapt_param_noise_async(self, batch):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('adapt_param_noise', batch))
        self.waiting = True

    def adapt_param_noise_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def train_async(self, batch):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('train', batch))
        self.waiting = True

    def train_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def get_stats_async(self, batch):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_stats', batch))
        self.waiting = True

    def get_stats_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecMAEnv after calling close()"


def _flatten_obs(obs):
    assert isinstance(obs, list) or isinstance(obs, tuple)
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        import collections
        assert isinstance(obs, collections.OrderedDict)
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

