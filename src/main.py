import random
import gymnasium as gym
import numpy as np

from learning_classes import Bot, GeneticAlgorithm

env = gym.make("LunarLander-v2", render_mode="") # remove render_mode="human" to run without visualization
observation, info = env.reset()
episode = 0

def run_random_agent():
    print("Go go go")
    for _ in range(5555):
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} finished with reward {reward} and info {info}")
            observation, info = env.reset()

    print("Terminating environment")
    env.close()

def run_genetic_algorithm():
    genetic_algorithm = GeneticAlgorithm(env, Bot)
    genetic_algorithm.run()
    print("Terminating environment")

run_genetic_algorithm()
