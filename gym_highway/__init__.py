import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Highway-v0',
    entry_point='gym_highway.envs:HighwayEnv'
    # kwargs={'manual': False, 'inf_obs': True, 'save': False, 'render': True}
)

# Multiagent envs
# ----------------------------------------
register(
    id='HighwayMultiagent-v0',
    entry_point='gym_highway.multiagent_envs:MultiAgentEnv'
)
