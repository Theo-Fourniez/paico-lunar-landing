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

def run_genetic_algorithm(env, bot, population_size=100, generations=1000, mutation_probability=0.15, survivor_number=30, new_crossover_bots=70):
    genetic_algorithm = GeneticAlgorithm(env, bot, population_size, generations, mutation_probability, survivor_number, new_crossover_bots)
    genetic_algorithm.run()
    return genetic_algorithm

def run_default_genetic_algorithm():
    genetic_algorithm = GeneticAlgorithm(env, Bot, generations=100, population_size=100, survivor_number=35, new_crossover_bots=65, mutation_probability=0.15)
    genetic_algorithm.run()
    return genetic_algorithm

def run_multiple_random_genetic_algorithm():
    random_id = random.randint(1, 100000)
    for i in range(500):
        genetic_algorithm = run_genetic_algorithm(env, Bot, population_size=100, generations=100, mutation_probability=0.25, survivor_number=30, new_crossover_bots=70)

def run_default_genetic_algorithm_with_json_bot(path):
    bot = Bot(env)
    bot.read_from_json(path)
    print(bot.weights)
    genetic_algorithm = GeneticAlgorithm(env, Bot, generations=100, population_size=100, survivor_number=35, new_crossover_bots=65, mutation_probability=0.15, starting_population=[bot])
    genetic_algorithm.run()
    return genetic_algorithm
#run_genetic_algorithm()
run_default_genetic_algorithm()
#run_default_genetic_algorithm_with_json_bot("best_bot_1709244959.402303.json")