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

def run_default_genetic_algorithm():
    genetic_algorithm = GeneticAlgorithm(env, Bot)
    genetic_algorithm.run()
    return genetic_algorithm

def run_multiple_genetic_algorithm():
    for i in range(500):
        print(i)
        genetic_algorithm = GeneticAlgorithm(env, Bot, save_results=True)
        genetic_algorithm.run()
    
def run_default_genetic_algorithm_with_json_bot(path):
    env = gym.make("LunarLander-v2", render_mode="human") # remove render_mode="human" to run without visualization
    env.reset()
    bot = Bot(env)
    bot.read_from_json(path)
    print(f"bot score {bot.score}")
    observation, reward, terminated, truncated, info = env.step(0)
    for _ in range(5555):
        bot.act(observation)
        observation, reward, terminated, truncated, info = bot.step()
        if terminated or truncated:
            print(f"Finished with reward {reward} and info {info}")
            observation, info = env.reset()
            return
    

def get_best_bot_from_ga(path):
    ga = GeneticAlgorithm(env, Bot)
    ga.read_from_json(path)
    print(ga.population)
    return ga.population[0]

#run_default_genetic_algorithm_with_json_bot('out/bot_18-03-2024_18:56:32_192.3.json')
run_multiple_genetic_algorithm()
#run_genetic_algorithm()
#run_default_genetic_algorithm()
#run_default_genetic_algorithm_with_json_bot("best_bot_1709244959.402303.json")