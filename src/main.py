import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()
episode = 0

print("Go go go")
for _ in range(5555):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        episode += 1
        print(f"Episode {episode} finished")
        observation, info = env.reset()

print("Terminating environment")
env.close()