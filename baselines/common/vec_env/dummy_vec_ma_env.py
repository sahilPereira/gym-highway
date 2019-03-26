import numpy as np
from gym import spaces
from . import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecMAEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        # env.observation_space is a list of n observations corresponding to n agents
        obs_space = env.observation_space

        # a normal obs return (None, shape, dtype)
        # list of observations contains obs for each agent from their perspective
        self.keys = []
        shapes = {}
        dtypes = {}
        for i in range(env.n):
            key, shape, dtype = obs_space_info(obs_space[i])
            self.keys.append(key)
            shapes.update(shape)
            dtypes.update(dtype)
        
        # buf_obs shape was originally (nenv, os_space), but now it is (nenv, n_agents, ob_space)
        # NOTE: the shape might be different if we have different observations per agent
        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(env.n, shapes[k]), dtype=dtypes[k]) for k in self.keys }

        # these all should handle multiple agents
        self.buf_dones = np.zeros((self.num_envs,env.n,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,env.n,), dtype=np.float32)

        self.buf_infos = [[{} for _ in range(env.n)] for _ in range(self.num_envs)]
        self.actions = None
        self.specs = [e.spec for e in self.envs]

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            if isinstance(self.envs[e].action_space, spaces.Discrete):
                action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)

            # if any agent in env is done, then reset env
            if any(self.buf_dones[e]):
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

