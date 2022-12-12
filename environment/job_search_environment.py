import functools
import random

import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple

from gymnasium.spaces.utils import flatten, flatdim

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.test import parallel_api_test

NUM_CANDIDATES = 3
NUM_EMPLOYERS = 3
EMPLOYER_BUDGET = 100
MAX_NUM_ITERS = 10

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
EMPLOYER_ACTIONS = [
    "NO_ACTION",
    "REJECT_APPLICANT",
    "MAKE_OFFER",
    "ACCEPT_COUNTER_OFFER",
    "REJECT_COUNTER_OFFER",
]
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
            agent: Tuple(
                (
                    Discrete(len(CANDIDATE_ACTIONS)),
                    Discrete(max(len(self._employers), len(self._candidates))),
                    Discrete(EMPLOYER_BUDGET + 1),
                    Discrete(MAX_NUM_ITERS + 1),
                )
            )
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: Dict(
                {
                    "job_openings": Dict(
                        {employer: Discrete(2) for employer in self._employers}
                    ),  # for each employer: 0 = not hiring, 1 = still hiring
                    "accepted_offer": Dict(
                        {
                            employer: Discrete(EMPLOYER_BUDGET + 1)
                            for employer in self._employers
                        }
                    ),
                    "current_offers": Dict(
                        {
                            employer: Tuple(
                                (
                                    Discrete(EMPLOYER_BUDGET + 1),
                                    Discrete(MAX_NUM_ITERS + 1),
                                )
                            )
                            for employer in self._employers
                        }
                    ),  # for each employer: (offer value, deadline); (0,0) = no offer
                    "rejected_offers": Dict(
                        {
                            employer: Tuple(
                                (Discrete(2), Discrete(EMPLOYER_BUDGET + 1))
                            )
                            for employer in self._employers
                        }
                    ),  # for each employer: 0 = not rejected/1 = rejected, value of rejected offer
                    "counter_offers": Dict(
                        {
                            employer: Tuple(
                                (
                                    Discrete(EMPLOYER_BUDGET + 1),
                                    Discrete(MAX_NUM_ITERS + 1),
                                )
                            )
                            for employer in self._employers
                        }
                    ),  # for each employer: (counter offer value, deadline)
                }
            )
            if "candidate" in agent
            else Dict(
                {
                    "candidate_strengths": Dict(
                        {
                            candidate: Discrete(MAX_CANDIDATE_STRENGTH + 1)
                            for candidate in self._candidates
                        }
                    ),  # Candidate strengths  (higher is better, will only be populated after a candidate applies)
                    "job_applicants": Dict(
                        {candidate: Discrete(2) for candidate in self._candidates}
                    ),  # 1 = applied
                    "outstanding_offers": Dict(
                        {
                            candidate: Tuple(
                                (
                                    Discrete(EMPLOYER_BUDGET + 1),
                                    Discrete(MAX_NUM_ITERS + 1),
                                )
                            )
                            for candidate in self._candidates
                        }
                    ),  # for each candidate: (offer value, deadline); (0,0) = no offer
                    "accepted_offers": Dict(
                        {candidate: Discrete(2) for candidate in self._candidates}
                    ),  # for each candidate: 0 = not declined/1 = declined, offer value of declined offer (declined by candidate)
                    "declined_offers": Dict(
                        {
                            candidate: Tuple(
                                (Discrete(2), Discrete(EMPLOYER_BUDGET + 1))
                            )
                            for candidate in self._candidates
                        }
                    ),  # for each candidate: 0 = not declined/1 = declined, offer value of declined offer (declined by candidate)
                    "counter_offers": Dict(
                        {
                            candidate: Tuple(
                                (
                                    Discrete(EMPLOYER_BUDGET + 1),
                                    Discrete(MAX_NUM_ITERS + 1),
                                )
                            )
                            for candidate in self._candidates
                        }
                    ),  # for each candidate: offer value from offer made by candidate, deadline
                    "rejected_offers": Dict(
                        {
                            candidate: Tuple(
                                (Discrete(2), Discrete(EMPLOYER_BUDGET + 1))
                            )
                            for candidate in self._candidates
                        }
                    ),  # for each candidate: 0 = not rejected/1 = rejected, offer value of counter offer that was rejected (rejected by employer)
                    "remaining_budget": Discrete(
                        EMPLOYER_BUDGET + 1
                    ),  # each employer will only have a budget of EMPLOYER_BUDGET
                }
            )
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
                    "current_offers": {
                        employer: (0, 0) for employer in self._employers
                    },
                    "rejected_offers": {
                        employer: (0, 0) for employer in self._employers
                    },
                    "counter_offers": {
                        employer: (0, 0) for employer in self._employers
                    },
                },
                "action_mask": np.zeros(flatdim(self.action_space(agent))),
            }
            if "candidate" in agent
            else {
                "observation": {
                    "candidate_strengths": {
                        candidate: 0 for candidate in self._candidates
                    },
                    "job_applicants": {candidate: 0 for candidate in self._candidates},
                    "outstanding_offers": {
                        candidate: (0, 0) for candidate in self._candidates
                    },
                    "accepted_offers": {candidate: 0 for candidate in self._candidates},
                    "declined_offers": {
                        candidate: (0, 0) for candidate in self._candidates
                    },
                    "counter_offers": {
                        candidate: (0, 0) for candidate in self._candidates
                    },
                    "rejected_offers": {
                        candidate: (0, 0) for candidate in self._candidates
                    },
                    "remaining_budget": EMPLOYER_BUDGET,
                },
                "action_mask": np.zeros(flatdim(self.action_space(agent))),
            }
            for agent in self.agents
        }

        self._candidate_stregnths = {
            candidate: random.randint(0, MAX_CANDIDATE_STRENGTH)
            for candidate in self._candidates
        }

        observations = self.game_state

        # Initialize action masks based on initial game_state
        self._update_action_masks()

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

        rewards = {agent: 0 for agent in self.agents}

        for agent in self.agents:
            action, target_index, new_offer_value, new_deadline = actions[agent]
            # print(action)
            # print(flatten(self.action_space(agent), (action, target_index, new_offer_value, new_deadline)))
            if "candidate" in agent:
                candidate = agent
                employer = f"employer_{target_index}"
                if action == NO_ACTION:
                    pass
                elif action == APPLY:
                    # Update employer observation
                    # Update job applicants
                    self.game_state[employer]["observation"]["job_applicants"][
                        candidate
                    ] = 1
                    # Update record of candidate strength
                    self.game_state[employer]["observation"]["candidate_strengths"][
                        candidate
                    ] = self._candidate_stregnths[agent]
                    # Update candidate observation
                    # Remove from job openings
                    self.game_state[candidate]["observation"]["job_openings"][
                        employer
                    ] = 0
                elif action == ACCEPT_OFFER:
                    # Get value of offer
                    offer_value, _ = self.game_state[employer]["observation"][
                        "outstanding_offers"
                    ][candidate]

                    # Update employer observations
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][
                        candidate
                    ] = (0, 0)
                    # Add to accepted offers
                    self.game_state[employer]["observation"]["accepted_offers"][
                        candidate
                    ] = 1
                    # NOTE: Decided to subtract offer value from the time offer is made
                    # # Reduce budget
                    # self.game_state[employer]["observation"]["remaining_budget"] -= offer_value

                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][
                        employer
                    ] = (0, 0)
                    # Add to accepted offer
                    self.game_state[candidate]["observation"]["accepted_offer"][
                        employer
                    ] = offer_value

                    # Update game state to reject all other offers that the candidate has
                    for e in self._employers:
                        if self.game_state[candidate]["observation"]["current_offers"][
                            e
                        ] != (0, 0):
                            # Add offer to employer declined overs
                            self.game_state[e]["observation"]["declined_offers"][
                                candidate
                            ] = self.game_state[e]["observation"]["outstanding_offers"][
                                candidate
                            ]
                            # Delete outstanding offer
                            self.game_state[e]["observation"]["outstanding_offers"][
                                candidate
                            ] = (0, 0)
                            # Offer from candidate's current offers
                            self.game_state[candidate]["observation"]["current_offers"][
                                e
                            ] = (0, 0)

                    # Update candidate reward
                    rewards[candidate] += offer_value / (
                        (1 + DISCOUNT_RATE) ** self.num_iters
                    )
                    # Update employer reward
                    rewards[employer] += (
                        self._candidate_stregnths[candidate] - offer_value
                    ) / ((1 + DISCOUNT_RATE) ** self.num_iters)
                elif action == REJECT_OFFER:
                    # Get value of offer
                    offer_value, _ = self.game_state[employer]["observation"][
                        "outstanding_offers"
                    ][candidate]

                    # Update employer observations
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][
                        candidate
                    ] = (0, 0)
                    # Add to declined offers
                    self.game_state[employer]["observation"]["declined_offers"][
                        candidate
                    ] = (1, offer_value)
                    # Add offer value back to budget
                    self.game_state[employer]["observation"][
                        "remaining_budget"
                    ] += offer_value

                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][
                        employer
                    ] = (0, 0)
                    # Add to rejected offers
                    self.game_state[candidate]["observation"]["rejected_offers"][
                        employer
                    ] = (1, offer_value)
                elif action == NEGOTIATE:
                    # Get value of offer
                    offer_value, _ = self.game_state[employer]["observation"][
                        "outstanding_offers"
                    ][candidate]
                    # Update employer observations
                    # Remove from outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][
                        candidate
                    ] = (0, 0)
                    # Add to counter offers
                    self.game_state[employer]["observation"]["counter_offers"][
                        candidate
                    ] = (new_offer_value, new_deadline)
                    # Add offer value back to budget
                    self.game_state[employer]["observation"][
                        "remaining_budget"
                    ] += offer_value

                    # Update candidate observations
                    # Remove from current offers
                    self.game_state[candidate]["observation"]["current_offers"][
                        employer
                    ] = (0, 0)
                    # Add to counter offers
                    self.game_state[candidate]["observation"]["counter_offers"][
                        employer
                    ] = (new_offer_value, new_deadline)
                else:
                    raise (ValueError, "Invalid candidate action")
            else:
                employer = agent
                candidate = f"candidate_{target_index}"
                if action == NO_ACTION:
                    pass
                elif action == REJECT_APPLICANT:
                    # Update employer observations
                    # Remove from applicants
                    self.game_state[employer]["observation"]["job_applicants"][
                        candidate
                    ] = 0
                    # Add to rejected offers with an offer value of 0 (offer never made)
                    self.game_state[employer]["observation"]["rejected_offers"][
                        candidate
                    ] = (1, 0)

                    # NOTE: No candidate observations to update
                elif action == MAKE_OFFER:
                    # NOTE: Instead of allowing offers to be made without subtracting from budget, instead subtract that amount from budget, and add back in if offer is declined, or offer expires

                    # Update employer observations
                    # Remove from applicants
                    self.game_state[employer]["observation"]["job_applicants"][
                        candidate
                    ] = 0
                    # Remove from counter offers (if applicable)
                    self.game_state[employer]["observation"]["counter_offers"][
                        candidate
                    ] = (0, 0)
                    # Update outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][
                        candidate
                    ] = (new_offer_value, new_deadline)
                    # Subtract offer value from remaining budget
                    self.game_state[employer]["observation"][
                        "remaining_budget"
                    ] -= new_offer_value

                    # Update candidate observations
                    # Add to current offers
                    self.game_state[candidate]["observation"]["current_offers"][
                        employer
                    ] = (new_offer_value, new_deadline)
                    # Remove from counter offers (if applicable)
                    self.game_state[candidate]["observation"]["counter_offers"][
                        employer
                    ] = (0, 0)
                elif action == ACCEPT_COUNTER_OFFER:
                    # Get offer value and deadline
                    offer_value, deadline = self.game_state[employer]["observation"][
                        "counter_offers"
                    ][candidate]

                    # Update employer observations
                    # Remove from counter offers
                    self.game_state[employer]["observation"]["counter_offers"][
                        candidate
                    ] = (0, 0)
                    # Update outstanding offers
                    self.game_state[employer]["observation"]["outstanding_offers"][
                        candidate
                    ] = (offer_value, deadline)
                    # Subtract offer value from remaining budget
                    self.game_state[employer]["observation"][
                        "remaining_budget"
                    ] -= new_offer_value

                    # Update candidate observations
                    # Remove from counter offers
                    self.game_state[candidate]["observation"]["counter_offers"][
                        employer
                    ] = (0, 0)
                    # Add to current offers
                    self.game_state[candidate]["observation"]["current_offers"][
                        employer
                    ] = (offer_value, deadline)
                elif action == REJECT_COUNTER_OFFER:
                    # Get offer value and deadline
                    offer_value, deadline = self.game_state[employer]["observation"][
                        "counter_offers"
                    ][candidate]

                    # Update employer observations
                    # Remove from counter offers
                    self.game_state[employer]["observation"]["counter_offers"][
                        candidate
                    ] = (0, 0)
                    # NOTE: employer rejects candidate, does NOT revert back to original offer
                    # Update rejected offers
                    self.game_state[employer]["observation"]["rejected_offers"][
                        candidate
                    ] = (1, offer_value)

                    # Update candidate observations
                    # Remove from counter offers
                    # self.game_state[candidate]["observation"]["counter_offers"][employer] = (0, 0)
                else:
                    raise (ValueError, "Invalid employer action")
            # Clean up all outstanding offers and counter offers that have expired
            # Move expired offers to declined/rejected field, as appropriate
            if "candidate" in agent:
                # Check candidate offers
                for e in self._employers:
                    # Remove all expired current offers
                    offer_value, deadline = self.game_state[candidate]["observation"][
                        "current_offers"
                    ][e]
                    if offer_value and deadline < self.num_iters:
                        self.game_state[candidate]["observation"]["current_offers"][
                            e
                        ] = (0, 0)
                        self.game_state[candidate]["observation"]["rejected_offers"][
                            e
                        ] = (1, offer_value)
                        self.game_state[e]["observation"]["outstanding_offers"][
                            candidate
                        ] = (0, 0)
                        self.game_state[e]["observation"]["declined_offers"][
                            candidate
                        ] = (1, offer_value)
                        self.game_state[e]["observation"][
                            "remaining_budget"
                        ] += new_offer_value
                    # Remove all expired counter offers
                    offer_value, deadline = self.game_state[candidate]["observation"][
                        "counter_offers"
                    ][e]
                    if offer_value and deadline < self.num_iters:
                        self.game_state[candidate]["observation"]["counter_offers"][
                            e
                        ] = (0, 0)
                        self.game_state[candidate]["observation"]["rejected_offers"][
                            e
                        ] = (1, offer_value)
                        self.game_state[e]["observation"]["counter_offers"][
                            candidate
                        ] = (0, 0)
                        self.game_state[e]["observation"]["rejected_offers"][
                            candidate
                        ] = (1, offer_value)
            else:
                for c in self._candidates:
                    # Remove expired outstanding offers
                    offer_value, deadline = self.game_state[employer]["observation"][
                        "outstanding_offers"
                    ][c]
                    if offer_value and deadline < self.num_iters:
                        self.game_state[c]["observation"]["current_offers"][
                            employer
                        ] = (0, 0)
                        self.game_state[c]["observation"]["rejected_offers"][
                            employer
                        ] = (1, offer_value)
                        self.game_state[employer]["observation"]["outstanding_offers"][
                            c
                        ] = (0, 0)
                        self.game_state[employer]["observation"]["declined_offers"][
                            c
                        ] = (1, offer_value)
                        self.game_state[employer]["observation"][
                            "remaining_budget"
                        ] += new_offer_value
                    # Remove expired counter offers
                    offer_value, deadline = self.game_state[employer]["observation"][
                        "counter_offers"
                    ][c]
                    if offer_value and deadline < self.num_iters:
                        self.game_state[c]["observation"]["counter_offers"][
                            employer
                        ] = (0, 0)
                        self.game_state[c]["observation"]["rejected_offers"][
                            employer
                        ] = (1, offer_value)
                        self.game_state[employer]["observation"]["counter_offers"][
                            c
                        ] = (0, 0)
                        self.game_state[employer]["observation"]["rejected_offers"][
                            c
                        ] = (1, offer_value)

        # Update all action masks based on updated observations/game state
        self._update_action_masks()

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
                terminations[agent] = any(
                    value != 0
                    for value in self.game_state[agent]["observation"][
                        "accepted_offer"
                    ].values()
                )
            else:
                terminations[agent] = self.game_state[agent]["observation"][
                    "remaining_budget"
                ] <= 0 or (
                    len(self._candidates)
                    == (
                        sum(
                            map(
                                lambda x: x == 1,
                                self.game_state[agent]["observation"][
                                    "accepted_offers"
                                ].values(),
                            )
                        )
                        + (
                            sum(
                                map(
                                    lambda x: x != (0, 0),
                                    self.game_state[agent]["observation"][
                                        "declined_offers"
                                    ].values(),
                                )
                            )
                        )
                        + (
                            sum(
                                map(
                                    lambda x: x != (0, 0),
                                    self.game_state[agent]["observation"][
                                        "rejected_offers"
                                    ].values(),
                                )
                            )
                        )
                    )
                )

        # Check truncation conditions (overwrites termination conditions)
        truncations = {agent: self.num_iters >= MAX_NUM_ITERS for agent in self.agents}
        self.num_iters += 1

        observations = self.game_state

        # Get dummy infos (not used)
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _update_action_masks(self):
        """Using the current game_state attribute, update action masks for each agent

        Action masks determine which actions are allowed

        Returns:
            None
        """
        candidate_employers_mask = np.concatenate(
            (
                np.ones(len(CANDIDATE_ACTIONS)),
                np.ones(len(self._employers)),
                np.zeros(
                    max(len(self._employers), len(self._candidates))
                    - len(self._employers)
                ),
                np.zeros(EMPLOYER_BUDGET + 1),
                np.zeros(MAX_NUM_ITERS + 1),
            )
        )

        employer_candidates_mask = np.concatenate(
            (
                np.ones(len(EMPLOYER_ACTIONS)),
                np.ones(len(self._candidates)),
                np.zeros(
                    max(len(self._employers), len(self._candidates))
                    - len(self._employers)
                ),
                np.zeros(EMPLOYER_BUDGET + 1),
                np.zeros(MAX_NUM_ITERS + 1),
            )
        )

        # Possible negotiating values
        def get_candidate_counter_offer_values_and_deadlines(
            current_offer_value, current_deadline
        ):
            # Candidate will counter with values strictly greater than current offer
            offer_values = np.concatenate(
                (
                    np.zeros(current_offer_value + 1),
                    np.ones(EMPLOYER_BUDGET + 1 - (current_offer_value + 1)),
                )
            )
            # Candidate will counter with deadline greater than or equal to current deadline
            deadlines = np.concatenate(
                (
                    np.zeros(current_deadline),
                    np.ones(MAX_NUM_ITERS + 1 - current_deadline),
                )
            )
            counter_offer_details = np.concatenate(
                (
                    np.zeros(len(CANDIDATE_ACTIONS)),
                    np.zeros(len(self._employers)),
                    offer_values,
                    deadlines,
                )
            )
            assert counter_offer_details.size == candidate_employers_mask.size
            return counter_offer_details

        def get_employer_offer_values_and_deadlines(
            candidate_strength,
            remaining_budget,
            counter_offer_value=(EMPLOYER_BUDGET + 1),
            counter_offer_deadline=self.num_iters,
        ):
            # Employer will only offer value weakly less than candidate strength or remaining budget, whichever is smaller
            offer_values = np.concatenate(
                (
                    np.ones(
                        min(candidate_strength, remaining_budget, counter_offer_value)
                        + 1
                    ),
                    np.zeros(
                        EMPLOYER_BUDGET
                        + 1
                        - (
                            min(
                                candidate_strength,
                                remaining_budget,
                                counter_offer_value,
                            )
                            + 1
                        )
                    ),
                )
            )
            # Employer will only offer deadline in the future
            deadlines = np.concatenate(
                (
                    np.zeros(counter_offer_deadline),
                    np.ones(MAX_NUM_ITERS + 1 - counter_offer_deadline),
                )
            )
            offer_details = np.concatenate(
                (
                    np.zeros(len(EMPLOYER_ACTIONS)),
                    np.zeros(len(self._candidates)),
                    offer_values,
                    deadlines,
                )
            )
            assert offer_details.size == employer_candidates_mask.size
            return offer_details

        for agent in self.agents:
            space = self.action_space(agent)
            action_mask = np.zeros(flatdim(space))
            if "candidate" in agent:
                for employer_index, employer in enumerate(self._employers):
                    # Job opening -> agent allowed to apply for all open jobs
                    if (
                        self.game_state[agent]["observation"]["job_openings"][employer]
                        == 1
                    ):
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (APPLY, employer_index, 0, 0)),
                                candidate_employers_mask,
                            ),
                        )
                    # Accepted offer -> no actions allowed
                    if (
                        self.game_state[agent]["observation"]["accepted_offer"][
                            employer
                        ]
                        != 0
                    ):
                        action_mask = np.zeros(flatdim(space))
                        break
                    if self.game_state[agent]["observation"]["current_offers"][
                        employer
                    ] != (0, 0):
                        current_offer_value, current_deadline = self.game_state[agent][
                            "observation"
                        ]["current_offers"][employer]
                        # Candidate can accept offer
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (ACCEPT_OFFER, employer_index, 0, 0)),
                                candidate_employers_mask,
                            ),
                        )
                        # Candidate can reject offer
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (REJECT_OFFER, employer_index, 0, 0)),
                                candidate_employers_mask,
                            ),
                        )
                        # Candidate can negotiate offer, with strictly higher offer value and weakly later deadline
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (NEGOTIATE, employer_index, 0, 0)),
                                candidate_employers_mask,
                            ),
                        )
                        counter_offer_details = (
                            get_candidate_counter_offer_values_and_deadlines(
                                current_offer_value, current_deadline
                            )
                        )
                        action_mask = np.logical_or(action_mask, counter_offer_details)
            else:
                remaining_budget = self.game_state[agent]["observation"][
                    "remaining_budget"
                ]
                for candidate_index, candidate in enumerate(self._candidates):
                    candidate_strength = self.game_state[agent]["observation"][
                        "candidate_strengths"
                    ][candidate]
                    # Job applicant -> agent can extend offer or reject candidate
                    if (
                        self.game_state[agent]["observation"]["job_applicants"][
                            candidate
                        ]
                        != 0
                    ):
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(
                                    space, (REJECT_APPLICANT, candidate_index, 0, 0)
                                ),
                                employer_candidates_mask,
                            ),
                        )
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (MAKE_OFFER, candidate_index, 0, 0)),
                                employer_candidates_mask,
                            ),
                        )

                        offer_details = get_employer_offer_values_and_deadlines(
                            candidate_strength, remaining_budget
                        )
                        action_mask = np.logical_or(action_mask, offer_details)
                    # If any counter offers exist
                    if self.game_state[agent]["observation"]["counter_offers"][
                        candidate
                    ] != (0, 0):
                        counter_offer_value, counter_offer_deadline = self.game_state[
                            agent
                        ]["observation"]["counter_offers"][candidate]
                        # Can only accept the counter offer if offer value is less than or equal to remaining budget
                        if counter_offer_value <= remaining_budget:
                            action_mask = np.logical_or(
                                action_mask,
                                np.logical_and(
                                    flatten(
                                        space,
                                        (ACCEPT_COUNTER_OFFER, candidate_index, 0, 0),
                                    ),
                                    employer_candidates_mask,
                                ),
                            )
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(
                                    space, (REJECT_COUNTER_OFFER, candidate_index, 0, 0)
                                ),
                                employer_candidates_mask,
                            ),
                        )
                        action_mask = np.logical_or(
                            action_mask,
                            np.logical_and(
                                flatten(space, (MAKE_OFFER, candidate_index, 0, 0)),
                                employer_candidates_mask,
                            ),
                        )
                        counter_offer_details = get_employer_offer_values_and_deadlines(
                            candidate_strength,
                            remaining_budget,
                            counter_offer_value,
                            counter_offer_deadline,
                        )
                        action_mask = np.logical_or(action_mask, counter_offer_details)
                    # No budget remaining -> no actions allowed
                    if remaining_budget == 0:
                        action_mask = np.zeros(flatdim(space))
                        break
            self.game_state[agent]["action_mask"] = action_mask.astype(int)


if __name__ == "__main__":
    parallel_api_test(JobSearchEnvironment(), num_cycles=1_000_000)
