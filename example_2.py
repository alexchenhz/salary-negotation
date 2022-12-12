import environment.job_search as environment
import random
import numpy as np

from gymnasium.spaces.utils import flatdim, unflatten

from environment.job_search import NUM_CANDIDATES, NUM_EMPLOYERS, EMPLOYER_BUDGET, MAX_NUM_ITERS, CANDIDATE_ACTIONS, EMPLOYER_ACTIONS

def sample_action(env, obs, agent):
    agent_obs = obs[agent]
    num_actions = len(CANDIDATE_ACTIONS) if "candidate" in agent else len(EMPLOYER_ACTIONS)
    num_targets = NUM_CANDIDATES if "candidate" in agent else NUM_EMPLOYERS
    # Check if action mask exists
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        action_mask = agent_obs["action_mask"]
        legal_actions = np.flatnonzero(action_mask[0:num_actions])
        legal_targets = np.flatnonzero(action_mask[num_actions: num_actions + num_targets])
        legal_offer_values = np.flatnonzero(action_mask[num_actions + num_targets: num_actions + num_targets + EMPLOYER_BUDGET + 1])
        legal_deadlines = np.flatnonzero(action_mask[num_actions + num_targets + EMPLOYER_BUDGET + 1: num_actions + num_targets + EMPLOYER_BUDGET + 1 + MAX_NUM_ITERS + 1])
        print(legal_actions)
        action = random.choice(legal_actions) if legal_actions.any() else 0 
        target = random.choice(legal_targets) if legal_targets.any() else 0
        offer_value = random.choice(legal_offer_values) if legal_offer_values.any() else 0
        deadline = random.choice(legal_deadlines) if legal_deadlines.any() else 0
        return action, target, offer_value, deadline
    return env.action_space(agent).sample()

if __name__ == "__main__":
    env = environment.env(render_mode="human")

    env.reset()

    # Initial testing

    # Apply to job 0
    act = {
        "candidate_0": (1, 0, 0, 0),
        "employer_0": (0, 0, 0, 0),
    }

    observations, rewards, terminations, truncations, infos = env.step(act)
    print(observations)
    
    while True:
        act = {
            "candidate_0": sample_action(env, observations, "candidate_0"),
            "employer_0": sample_action(env, observations, "employer_0")
        }
        print(act)

        observations, rewards, terminations, truncations, infos = env.step(act)
        print(observations)
        if any([terminations[agent] for agent in env.agents]) or any([truncations[agent] for agent in env.agents]):
            print(rewards)
            break
    

    # # Make offer to candidate 0
    # act = {
    #     "candidate_0": (0, 0, 0, 0),
    #     "employer_0": (2, 0, 40, 4),
    # }

    # observations, rewards, terminations, truncations, infos = env.step(act)
    # print(observations, rewards, terminations)

    # print(sample_action(env, observations, "candidate_0"))
    # # # Accept offer
    # # act = {
    # #     "candidate_0": (2, 0, 0, 0),
    # #     "employer_0": (0, 0, 0, 0),
    # # }

    # # observations, rewards, terminations, truncations, infos = env.step(act)
    # # print(observations, rewards, terminations)

    # # # Reject offer
    # # act = {
    # #     "candidate_0": (3, 0, 0, 0),
    # #     "employer_0": (0, 0, 0, 0),
    # # }

    # # observations, rewards, terminations, truncations, infos = env.step(act)
    # # print(observations, rewards, terminations)

    # # Counter offer
    # act = {
    #     "candidate_0": (4, 0, 60, 8),
    #     "employer_0": (0, 0, 0, 0),
    # }

    # observations, rewards, terminations, truncations, infos = env.step(act)
    # print(observations, rewards, terminations)

    # # Accept counter offer
    # act = {
    #     "candidate_0": (0, 0, 0, 0),
    #     "employer_0": (3, 0, 0, 0),
    # }

    # observations, rewards, terminations, truncations, infos = env.step(act)
    # print(observations, rewards, terminations)

    # # Accept offer
    # act = {
    #     "candidate_0": (2, 0, 0, 0),
    #     "employer_0": (0, 0, 0, 0),
    # }

    # observations, rewards, terminations, truncations, infos = env.step(act)
    # print(observations, rewards, terminations)