import environment.job_search as environment
import random

env = environment.env(render_mode="human")

env.reset()

# for agent in env.agent_iter():
#     print(agent)
#     observation, reward, terminated, truncated, info = env.last()
#     if terminated or truncated:
#         env.close()
#         act = None
#     else:
#         act = {
#             "candidate_0": (0, 0, 0, 0),
#             "employer_0": (0, 0, 0, 0),
#         }
#     env.step(act)

# observation, reward, terminated, truncated, info = env.last()
# if terminated or truncated:
#     env.close()
#     act = None
# else:
act = {
    "candidate_0": (1, 0, 0, 0),
    "employer_0": (0, 0, 0, 0),
}

observations, rewards, terminations, truncations, infos = env.step(act)
print(observations)