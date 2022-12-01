import environment.job_search as environment
import random

env = environment.env(render_mode="human")

env.reset()

for agent in env.agent_iter():
    print(agent)
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.close()
        act = None
    else:
        act = 0
    print(agent, observation, reward)
    env.step(act)
