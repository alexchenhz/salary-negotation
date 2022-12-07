import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Text

from gymnasium.spaces.utils import flatten, flatdim

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


NUM_CANDIDATES = 1
NUM_EMPLOYERS = 1
EMPLOYER_BUDGET = 100
MAX_NUM_ITERS = 100

NO_ACTION = 0
APPLY = 1
ACCEPT_OFFER = 2
REJECT_OFFER = 3
NEGOTIATE = 4
CANDIDATE_ACTIONS = ["NO_ACTION", "APPLY", "ACCEPT_OFFER", "REJECT_OFFER", "NEGOTIATE"]

REJECT_APPLICANT = 1
MAKE_OFFER = 2
ACCEPT_COUNTER_OFFER = 3
REJECT_COUNTER_OFFER = 4
EMPLOYER_ACTIONS = ["NO_ACTION", "REJECT_APPLICANT", "MAKE_OFFER", "ACCEPT_COUNTER_OFFER", "REJECT_COUNTER_OFFER"]
MAX_CANDIDATE_STRENGTH = 100

DISCOUNT_RATE = 0.05

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
    # env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = JobSearchEnvironment(render_mode=render_mode)
    # env = parallel_to_aec(env)
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
                Tuple((
                    Discrete(4), 
                    Discrete(len(self._employers)), 
                    Discrete(EMPLOYER_BUDGET + 1), 
                    Discrete(MAX_NUM_ITERS + 1)
                )) if "candidate" in agent else
                Tuple((
                    Discrete(4),
                    Discrete(len(self._candidates)),
                    Discrete(EMPLOYER_BUDGET + 1), 
                    Discrete(MAX_NUM_ITERS + 1)
                ))
            for agent in self.possible_agents
        }
        # self._action_spaces = {
        #     agent:
        #         Dict({
        #             "apply_to_job": Discrete(len(self._employers)), # index of the employer
        #             "accept_offer": Discrete(len(self._employers)), # index of the employer
        #             "negotiate_offer": Tuple((Discrete(len(self._employers)), Discrete(100), Discrete(100))), # (Index of the employer, 
        #             "reject_offer": Discrete(len(self._employers))
        #         }) if "candidate" in agent else
        #         Dict({
        #             "reject_applicant": Discrete(len(self._candidates)),
        #             "make_offer": Tuple((Discrete(len(self._candidates)), Discrete(100), Discrete(100))),
        #             "accept_counter_offer": Discrete(len(self._candidates)),
        #             "reject_counter_offer": Discrete(len(self._candidates))
        #         })
        #     for agent in self.possible_agents
        # }
        
        # Define observation spaces
        self._observation_spaces = {
            agent: 
                Dict(
                {
                    "job_openings": Dict({employer: Discrete(2)for employer in self._employers}), # for each employer: 0 = not hiring, 1 = still hiring
                    "accepted_offer": Dict({employer: Discrete(EMPLOYER_BUDGET + 1) for employer in self._employers}),
                    "current_offers": Dict({employer: Tuple((Discrete(EMPLOYER_BUDGET + 1), Discrete(MAX_NUM_ITERS + 1))) for employer in self._employers}), # for each employer: (offer value, deadline); (0,0) = no offer
                    "rejected_offers": Dict({employer: Tuple((Discrete(2), Discrete(EMPLOYER_BUDGET + 1))) for employer in self._employers}), # for each employer: 0 = not rejected/1 = rejected, value of rejected offer
                    "counter_offers": Dict({employer: Tuple((Discrete(EMPLOYER_BUDGET + 1), Discrete(MAX_NUM_ITERS + 1))) for employer in self._employers}), # for each employer: (counter offer value, deadline)
                }) if "candidate" in agent else
                Dict(
                {
                    "job_applicants": Dict({candidate: Tuple((Discrete(2), Discrete(MAX_CANDIDATE_STRENGTH + 1))) for candidate in self._candidates}), # (1 = applied, 0-9 = strength of candidate higher is better)
                    "outstanding_offers": Dict({candidate: Tuple((Discrete(EMPLOYER_BUDGET + 1), Discrete(MAX_NUM_ITERS + 1))) for candidate in self._candidates}), # for each candidate: (offer value, deadline); (0,0) = no offer
                    "accepted_offers": Dict({candidate: Discrete(2) for candidate in self._candidates}), # for each candidate: 0 = not declined/1 = declined, offer value of declined offer (declined by candidate)
                    "declined_offers": Dict({candidate: Tuple((Discrete(2), Discrete(EMPLOYER_BUDGET + 1))) for candidate in self._candidates}), # for each candidate: 0 = not declined/1 = declined, offer value of declined offer (declined by candidate)
                    "counter_offers": Dict({candidate: Tuple((Discrete(EMPLOYER_BUDGET + 1), Discrete(MAX_NUM_ITERS + 1))) for candidate in self._candidates}), # for each candidate: offer value from offer made by candidate, deadline
                    "rejected_offers": Dict({candidate: Tuple((Discrete(2), Discrete(EMPLOYER_BUDGET + 1))) for candidate in self._candidates}), # for each candidate: 0 = not rejected/1 = rejected, offer value of counter offer that was rejected (rejected by employer)
                    "remaining_budget": Discrete(EMPLOYER_BUDGET + 1), # each employer will only have a budget of EMPLOYER_BUDGET
                })
            for agent in self.possible_agents
        }
        
        self.game_state = None
        self.candidate_stregnths = None
    
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
        self.num_iters = 0
        
        # Game state is the same as all of the observations for each agent
        self.game_state = {
            agent: { 
                "observation": {
                    "job_openings": {employer: 1 for employer in self._employers},
                    "accepted_offer": {employer: 0 for employer in self._employers},
                    "current_offers": {employer: (0, 0) for employer in self._employers},
                    "rejected_offers": {employer: (0, 0) for employer in self._employers},
                    "counter_offers": {employer: (0, 0) for employer in self._employers}
                },
                # FIXME: Need to allow apply action with employer index (1's for first entry and all employer index entries)
                # Probably should just create helper functions to handle to action_mask changes
                # Use np.logical_or.reduce([np_arrays created with flatten and slicing])
                "action_mask": np.zeros(flatdim(self.action_space(agent)))
            } if "candidate" in agent else
            {
                "observation": {
                    "job_applicants": {candidate: (0, 0) for candidate in self._candidates},
                    "outstanding_offers": {candidate: (0, 0) for candidate in self._candidates},
                    "accepted_offers": {candidate: 0 for candidate in self._candidates},
                    "declined_offers": {candidate: (0, 0) for candidate in self._candidates},
                    "counter_offers": {candidate: (0, 0) for candidate in self._candidates},
                    "rejected_offers": {candidate: (0, 0) for candidate in self._candidates},
                    "remaining_budget": EMPLOYER_BUDGET,
                },
                "action_mask": np.zeros(flatdim(self.action_space(agent)))
            }
            for agent in self.agents
        }
        
        self._candidate_stregnths = {candidate: random.randint(0, MAX_CANDIDATE_STRENGTH) for candidate in self._candidates}
        
        observations = self.game_state
        
        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos
        
    def step(self, actions):        
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        print(actions)

        # Execute actions
        candidate_actions = [actions[agent] for agent in self._candidates]
        employer_actions = [actions[agent] for agent in self._employers]
        
        rewards = {agent: 0 for agent in self.agents}
        
        for agent in self.agents:
            action, target_index, new_offer_value, new_deadline = actions[agent]
            print("action:", action)
            if "candidate" in agent:
                candidate = agent
                employer = f"employer_{target_index}"
                if action == NO_ACTION:
                    pass
                elif action == APPLY:
                    # Update employer observation
                    self.game_state[employer]["observation"]["job_applicants"][candidate] = (0, self._candidate_stregnths[agent])
                    # Update candidate observation
                    self.game_state[candidate]["observation"]["job_openings"][employer] = 0
                    # TODO: Action mask -> cannot apply to this same employer
                    # TODO: One difference between the game state and observations: candidate might not be able to see their own strength
                elif action == ACCEPT_OFFER:
                    # Get value of offer
                    _, offer_value = self.game_state[employer]["observation"]["outstanding_offers"][candidate]
                    
                    # Update employer observations
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][candidate] = (0, 0)
                    # Add to accepted offers
                    self.game_state[employer]["observation"]["accepted_offers"][candidate] = 1
                    # Reduce budget
                    self.game_state[employer]["observation"]["remaining_budget"] -= offer_value
                    
                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][employer] = (0, 0)
                    # Add to accepted offer
                    self.game_state[candidate]["observation"]["accepted_offer"][employer] = offer_value
                    
                    # Update game state to reject all other offers that the candidate has
                    for e in self._employers:
                        if self.game_state[candidate]["observation"]["current_offers"][e] != (0, 0):
                            # Add offer to employer declined overs
                            self.game_state[e]["observation"]["declined_offers"][candidate] = self.game_state[e]["observation"]["outstanding_offers"][candidate]
                            # Delete outstanding offer
                            self.game_state[e]["observation"]["outstanding_offers"][candidate] = (0, 0)
                            # Offer from candidate's current offers
                            self.game_state[candidate]["observation"]["current_offers"][e] = (0, 0)
                    # Update action mask -> this agent is done, so no remaining actions
                    candidate_action_mask = np.zeros(flatdim(self.action_space(candidate)))
                    
                    # Update candidate reward
                    rewards[candidate] += offer_value / ((1 + DISCOUNT_RATE) ** self.num_iters)
                    # Update employer reward
                    rewards[employer] += (self._candidate_stregnths[candidate] - offer_value) / ((1 + DISCOUNT_RATE) ** self.num_iters)
                elif action == REJECT_OFFER:
                    # Get value of offer
                    _, offer_value = self.game_state[employer]["observation"]["outstanding_offers"][candidate]
                    
                    # Update employer observations
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][candidate] = (0, 0)
                    # Add to rejected offers
                    self.game_state[employer]["observation"]["declined_offers"][candidate] = (1, offer_value)
                    
                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][employer] = (0, 0)
                    # Add to rejected offers
                    self.game_state[candidate]["observation"]["rejected_offers"][employer] = (1, offer_value)
                elif action == NEGOTIATE:
                    # Update employer observations
                    self.game_state[employer]["observation"]["rejected_offers"][candidate] = (1, offer_value)
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][candidate] = (0, 0)
                    # Add to counter offers
                    self.game_state[employer]["observation"]["counter_offers"][candidate] = (new_offer_value, new_deadline)
                    
                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][employer] = (0, 0)
                    # Add to counter offers
                    self.game_state[candidate]["observation"]["counter_offers"][employer] = (new_offer_value, new_deadline)
                else:
                    raise(ValueError, "Invalid candidate action")
            else:
                employer = agent
                candidate = f"candidate_{target_index}"
                if action == NO_ACTION:
                    pass
                elif action == REJECT_APPLICANT:
                    # Update employer observations
                    # Remove from applicants
                    self.game_state[employer]["observation"]["job_applicants"][candidate] = (0, 0)
                    # Add to rejected offers
                    self.game_state[employer]["observation"]["rejected_offers"][candidate] = (1, 0) # TODO: Is this the best implementation of rejection? Do employers even need to keep information regarding the offer that was rejected?

                    # NOTE: No candidate observations to update
                elif action == MAKE_OFFER:
                    # Update employer observations
                    # Remove from applicants
                    self.game_state[employer]["observation"]["job_applicants"][candidate] = (0, 0)
                    # Update outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][candidate] = (new_offer_value, new_deadline)
                    
                    # Update candidate observations
                    # Add to current offers
                    self.game_state[candidate]["observation"]["current_offers"][candidate] = (new_offer_value, new_deadline)
                elif action == ACCEPT_COUNTER_OFFER:
                    # Update employer observations
                    # Remove from counter offers
                    self.game_state[employer]["observation"]["counter_offers"][candidate] = (0, 0)
                    # Update outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][candidate] = (new_offer_value, new_deadline)
                    
                    # Update candidate observations
                    # Add to current offers
                    self.game_state[candidate]["observation"]["current_offers"][candidate] = (new_offer_value, new_deadline)
                elif action == REJECT_COUNTER_OFFER:
                    # Update employer observations
                    # Remove from counter offers
                    self.game_state[employer]["observation"]["counter_offers"][candidate] = (0, 0)
                    # TODO: Should employer reject? Or revert to offering the original outstanding offer?
                    # Update rejected offers
                    self.game_state[employer]["observation"]["rejected_offers"][candidate] = (new_offer_value, new_deadline)
                    
                    # Update candidate observations
                    # Remove from counter offers
                    self.game_state[candidate]["observation"]["counter_offers"][candidate] = (0, 0)
                else:
                    raise(ValueError, "Invalid employer action")
            # TODO: After each iteration, update action masks based on observations for employer/candidate
            
            # TODO: Clean up all outstanding offers that have expired, and update action mask as appropriate
            # Check candidate offers
            for e in self._employers:
                _, deadline = self.game_state[candidate]["observation"]["current_offers"][e]
                if deadline < self.num_iters:
                    self.game_state[candidate]["observation"]["current_offers"][e] = (0, 0)
            for c in self._candidates:
                _, deadline = self.game_state[employer]["observation"]["counter_offers"][c]
                if deadline < self.num_iters:
                    self.game_state[employer]["observation"]["counter_offers"][c] = (0, 0)
                   
                
        """
        Check termination conditions 
        
        1. For candidates, terminate when offer is accepted (note, candidates 
        do not know when they have been rejected (the classic ghosted rejection))
        
        2. For employers, terminate when no budget remaining OR all candidates 
        have either accepted an offer, declined an offer, or had their counter offer 
        rejected
        """
        terminations = {}
        for agent in self.agents:
            if "candidate" in agent:
                terminations[agent] = any(value != 0 for value in self.game_state[candidate]["observation"]["accepted_offer"].values())
            else:
                terminations[agent] = self.game_state[employer]["observation"]["remaining_budget"] <= 0 or (
                    len(self._candidates) == 
                        (sum(map(lambda x: x == 1, self.game_state[employer]["observation"]["accepted_offers"].values()))
                        + (sum(map(lambda x: x != (0,0), self.game_state[employer]["observation"]["declined_offers"].values())))
                        + (sum(map(lambda x: x != (0,0), self.game_state[employer]["observation"]["rejected_offers"].values())))
                    ))
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {
                agent: self.num_iters >= MAX_NUM_ITERS for agent in self.agents
            }
        self.num_iters += 1
        
        observations = self.game_state
        
        # Get dummy infos (not used)
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos