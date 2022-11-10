import environment
import random

env = environment.env(render_mode="human")

env.reset()

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.close()
        act = None
    else:
        act = random.choice([0,1,2])
    print(agent, observation, reward)
    env.step(act)
