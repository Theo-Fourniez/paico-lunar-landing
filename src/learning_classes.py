# a genetic algorithm bot class 
import random
import numpy as np
import gymnasium as gym

class Bot:
    def __init__(self, env: gym.Env):
        self.env = env
        self.observation = env.reset()
        self.action = 0
        self.weights = np.random.rand(8, 4) # 8 observations, 4 actions TODO : change this to be dynamic

    def act(self, observation):
        result_action_matrix = np.dot(observation, self.weights)
        self.action = np.argmax(result_action_matrix) 
        return self.action

    def step(self):
        observation, reward, terminated, truncated, info = self.env.step(self.action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.observation = self.env.reset()
        return self.observation
    
# a genetic algorithm class
class GeneticAlgorithm:
    def __init__(self, env, bot_class: Bot, population_size=100, generations=100):
        self.env = env
        self.bot_class = bot_class
        self.population_size = population_size
        self.generations = generations
        self.population = [self.bot_class(self.env) for _ in range(self.population_size)]
        self.scores = [0 for _ in range(self.population_size)]

    def run(self):
        for generation in range(self.generations):
            print(f"Running generation {generation}")
            for i in range(self.population_size):
                print(f"Running bot {i}")
                bot = self.population[i]
                for _ in range(50000):
                    print(f"Generation {generation}, bot {i}, step {_}")
                    observation, reward, terminated, truncated, info = bot.step()
                    action = bot.act(observation)
                    self.scores[i] += reward
                    if terminated or truncated:
                        break
                bot.reset()
            print(f"Generation {generation} finished")
            self.evolve()
        self.env.close()
    def crossover(self, bot1, bot2):
        bot = self.bot_class(self.env)
        for i in range(len(bot1.weights)):
            bot.weights[i] = bot1.weights[i] if random.random() > 0.5 else bot2.weights[i]
        return bot 
    def evolve(self):
        MUTATION_PROBABILITY = 0.1
        # select the 90% best bots
        best_bots = [self.population[i] for i in sorted(range(self.population_size), key=lambda i: self.scores[i], reverse=True)[:int(self.population_size * 0.9)]]
        # crossover the best bots to create the next generation
        self.population = best_bots + [self.crossover(best_bots[i], best_bots[j]) for i in range(len(best_bots)) for j in range(i + 1, len(best_bots))]
        # mutate the genes of the next generation
        for bot in self.population:
            for i in range(len(bot.weights)):
                if random.random() < MUTATION_PROBABILITY:
                    bot.weights[i] = bot.weights[i] + np.random.rand()