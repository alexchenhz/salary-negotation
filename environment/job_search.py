import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Text

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    """
    This wrapper helps error handling for discrete action spaces
    
    NOTE: see pettingzoo/utils/wrappers/assert_out_of_bounds.py
    This checks that every agent in self.possible_agents is a Discrete agent.
    This assertion fails because possible_agents is a list of strings, with a 
    mapping of those strings to Discrete agent objects.
    """
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class JobSearchEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "js_v0"}
    
    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        # Create candidate and employer agents
        self._candidates = ["candidate_" + str(r) for r in range(NUM_CANDIDATES)]
        self._employers = ["employer_" + str(r) for r in range(NUM_EMPLOYERS)]
        self.possible_agents = self._candidates + self._employers
        
        # Map agent names to numbers, 0 through (number of agents) - 1
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
    