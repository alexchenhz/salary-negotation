import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Text

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers


NUM_CANDIDATES = 2
NUM_EMPLOYERS = 2
EMPLOYER_BUDGET = 1000

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
        
        # Define action spaces
        self._action_spaces = {
            agent:
                Dict({
                    "apply_to_job": Discrete(len(self._employers)), # index of the employer
                    "accept_offer": Discrete(len(self._employers)), # index of the employer
                    "negotiate_offer": Tuple((Discrete(len(self._employers)), Discrete(100), Discrete(100))), # (Index of the employer, 
                    "reject_offer": Discrete(len(self._employers))
                }) if "candidate" in agent else \
                Dict({
                    "reject_applicant": Discrete(len(self._candidates)),
                    "make_offer": Tuple((Discrete(len(self._candidates)), Discrete(100), Discrete(100))),
                    "accept_counter_offer": Discrete(len(self._candidates)),
                    "reject_counter_offer": Discrete(len(self._candidates))
                })
            for agent in self.possible_agents
        }
        
        # Define observation spaces
        self._observation_spaces = {
            agent: 
                Dict(
                {
                    "job_openings": Dict({employer: Discrete(2)for employer in self._employers}), # for each employer: 0 = not hiring, 1 = still hiring
                    "current_offers": Dict({employer: Tuple((Discrete(100), Discrete(100))) for employer in self._employers}), # for each employer: (offer value, deadline); (0,0) = no offer
                    "rejected_offers": Dict({employer: Tuple((Discrete(2), Discrete(100))) for employer in self._employers}), # for each employer: 0 = not rejected/1 = rejected, value of rejected offer
                    "counter_offers": Dict({employer: Tuple((Discrete(100), Discrete(100))) for employer in self._employers}), # for each employer: (counter offer value, deadline)
                }) if "candidate" in agent else
                Dict(
                {
                    "job_applicants": Dict({candidate: Tuple((Discrete(2), Discrete(10))) for candidate in self._candidates}), # (1 = applied, 0-9 = strength of candidate higher is better)
                    "outstanding_offers": Dict({candidate: Tuple((Discrete(100), Discrete(100))) for candidate in self._candidates}), # for each candidate: (offer value, deadline); (0,0) = no offer
                    "declined_offers": Dict({candidate: Tuple((Discrete(2), Discrete(100))) for candidate in self._candidates}), # for each candidate: 0 = not declined/1 = declined, offer value of declined offer (declined by candidate)
                    "counter_offers": Dict({candidate: Tuple((Discrete(100), Discrete(100))) for candidate in self._candidates}), # for each candidate: offer value from offer made by candidate, deadline
                    "rejected_offers": Dict({candidate: Tuple((Discrete(2), Discrete(100))) for candidate in self._candidates}), # for each candidate: 0 = not rejected/1 = rejected, offer value of counter offer that was rejected (rejected by employer)
                    "remaining_budget": Discrete(EMPLOYER_BUDGET + 1), # each employer will only have a budget of EMPLOYER_BUDGET
                })
            for agent in self.possible_agents
        }
        
        self.state = None
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        # TODO: should this info be in the environment, or in the main file?
        self.state = {
            agent: {
                "job_openings": {employer: 0 for employer in self._employers},
                "current_offers": {employer: (0, 0) for employer in self._employers},
                "rejected_offers": {employer: (0, 0) for employer in self._employers},
                "counter_offers": {employer: (0, 0) for employer in self._employers}
            } if "candidate" in agent else
            {
                "job_applicants": {candidate: (0, 0) for candidate in self._candidates},
                "outstanding_offers": {candidate: (0, 0) for candidate in self._candidates},
                "declined_offers": {candidate: (0, 0) for candidate in self._candidates},
                "counter_offers": {candidate: (0, 0) for candidate in self._candidates},
                "rejected_offers": {candidate: (0, 0) for candidate in self._candidates},
                "remaining_budget": EMPLOYER_BUDGET,
            }
            for agent in self.possible_agents
        }