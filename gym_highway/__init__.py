import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Highway-v0',
    entry_point='gym_highway.envs:HighwayEnv',
    kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
)