import environment.job_search as environment
import random

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

# Make offer to candidate 0
act = {
    "candidate_0": (0, 0, 0, 0),
    "employer_0": (2, 0, 40, 4),
}

observations, rewards, terminations, truncations, infos = env.step(act)
print(observations, rewards, terminations)

# # Accept offer
# act = {
#     "candidate_0": (2, 0, 0, 0),
#     "employer_0": (0, 0, 0, 0),
# }

# observations, rewards, terminations, truncations, infos = env.step(act)
# print(observations, rewards, terminations)

# # Reject offer
# act = {
#     "candidate_0": (3, 0, 0, 0),
#     "employer_0": (0, 0, 0, 0),
# }

# observations, rewards, terminations, truncations, infos = env.step(act)
# print(observations, rewards, terminations)

# Counter offer
act = {
    "candidate_0": (4, 0, 60, 8),
    "employer_0": (0, 0, 0, 0),
}

observations, rewards, terminations, truncations, infos = env.step(act)
print(observations, rewards, terminations)

# Accept counter offer
act = {
    "candidate_0": (0, 0, 0, 0),
    "employer_0": (3, 0, 0, 0),
}

observations, rewards, terminations, truncations, infos = env.step(act)
print(observations, rewards, terminations)

# Accept offer
act = {
    "candidate_0": (2, 0, 0, 0),
    "employer_0": (0, 0, 0, 0),
}

observations, rewards, terminations, truncations, infos = env.step(act)
print(observations, rewards, terminations)