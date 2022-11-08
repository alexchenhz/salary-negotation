import environment

env = environment.env(render_mode="human")

env.reset()

for agent in env.agent_iter():
    print(agent)
    act = 0
    x = env.step(act)
    y = env.last()
    print("output", x, y)
    observation, reward, terminated, truncated, info = env.last()
    
    if terminated or truncated:
        env.close()
        break