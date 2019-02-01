import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_bonuses = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        epbonuses = np.zeros(self.env.num_envs, dtype=np.float32)
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

            # get predictor bonus
            if self.model.unsup:
                # convert actions to one hot coding
                numaction = self.env.action_space.n
                one_hot_actions = np.eye(numaction)[actions]
                one_hot_actions = np.array(one_hot_actions, dtype=np.float32)

                bonuses = self.model.pred_bonuses(mb_obs[-1], self.obs, one_hot_actions)
                mb_bonuses.append(bonuses)
                epbonuses += bonuses
            
            # for info in infos:
            for i in range(len(infos)):
                info = infos[i]
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    # TODO: this method of tracking the bonus is not accurate, should be changed in the future
                    # NOT sure if this is fixed now, need to test
                    if self.model.unsup:
                        # add bonuses to epinfos
                        maybeepinfo.update({'bonus':epbonuses[i]})
                        # reset epbonuses for next steps
                        epbonuses[i] = 0.0
                    epinfos.append(maybeepinfo)
        
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        # last values for last obs
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # add prediction bonuses to original rewards
        if self.model.unsup:
            mb_bonuses = np.asarray(mb_bonuses, dtype=np.float32)
            mb_rewards += mb_bonuses
        
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1

    converts (nsteps, nenv, nfeatures) -> (nsteps*nenv, nfeatures) or (nsteps, nfeatures) -> (nsteps*nfeatures, )
    example: (240, 6, 16) -> (1440, 16)
             (240, 6) -> (1440,)
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

