from abc import ABC, abstractmethod

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecTrainer(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    def __init__(self, num_trainers):
        self.num_trainers = num_trainers
        # self.observation_space = observation_space
        # self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, obs, apply_noise=True, compute_Q=False):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def generate_index_async(self):
        pass

    @abstractmethod
    def generate_index_wait(self):
        pass

    @abstractmethod
    def sample_batch_async(self, replay_sample_index):
        pass

    @abstractmethod
    def sample_batch_wait(self):
        pass

    @abstractmethod
    def initialize_async(self, sess):
        pass

    @abstractmethod
    def initialize_wait(self):
        pass

    @abstractmethod
    def agent_initialize_async(self, sess):
        pass

    @abstractmethod
    def agent_initialize_wait(self):
        pass

    @abstractmethod
    def store_transition_async(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
        pass

    @abstractmethod
    def store_transition_wait(self):
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

# trainers[0].load(load_path)
# distance = agent.adapt_param_noise(trainers)
# cl, al = agent.train(trainers)
# stats = agent.get_stats(trainers)
# trainers[0].save(savepath)

    def initialize(self, sess):
        """
        Initialize the trainers synchronously.

        This is available for backwards compatibility.
        """
        self.initialize_async(sess)
        return self.initialize_wait()

    def agent_initialize(self, sess):
        """
        agent_initialize the trainers synchronously.

        This is available for backwards compatibility.
        """
        self.agent_initialize_async(sess)
        return self.agent_initialize_wait()
    
    def step(self, obs, apply_noise=True, compute_Q=False):
        """
        Step the trainers synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(obs, apply_noise=apply_noise, compute_Q=compute_Q)
        return self.step_wait()

    def store_transition(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
        self.store_transition_async(obs_n, actions_n, rew_n, new_obs_n, done_n)
        return self.store_transition_wait()

    def generate_index(self):
        self.generate_index_async()
        return self.generate_index_wait()

    def sample_batch(self, replay_sample_index):
        self.sample_batch_async(replay_sample_index)
        return self.sample_batch_wait()

    # def adapt_param_noise(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
    #     self.adapt_param_noise_async(obs_n, actions_n, rew_n, new_obs_n, done_n)
    #     return self.adapt_param_noise_wait()

    # def train(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
    #     self.train_async(obs_n, actions_n, rew_n, new_obs_n, done_n)
    #     return self.train_wait()

    # def get_stats(self, obs_n, actions_n, rew_n, new_obs_n, done_n):
    #     self.get_stats_async(obs_n, actions_n, rew_n, new_obs_n, done_n)
    #     return self.get_stats_wait()

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
                        num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
