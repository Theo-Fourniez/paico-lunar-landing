# a genetic algorithm bot class 
import random
import time
import numpy as np
import gymnasium as gym

class Bot:
    def __init__(self, env: gym.Env):
        self.env = env
        self.action = 0
        self.weights = np.random.rand(8, 4) # 8 observations, 4 actions TODO : change this to be dynamic

    def act(self, observation):
        result_action_matrix = observation @ self.weights # matrix multiplication
        self.action = np.argmax(result_action_matrix) 
        return self.action

    def step(self):
        observation, reward, terminated, truncated, info = self.env.step(self.action)
        return observation, reward, terminated, truncated, info
    
# a genetic algorithm class
class GeneticAlgorithm:
    def __init__(self, env, bot_class: Bot, population_size=100, generations=1000):
        self.env = env
        self.bot_class = bot_class
        self.population_size = population_size
        self.generations = generations
        self.population = [self.bot_class(self.env) for _ in range(self.population_size)]
        self.scores = [0 for _ in range(self.population_size)]

    def run(self):
        print(f"Running genetic algorithm with {self.population_size} bots for {self.generations} generations.")
        for generation in range(self.generations):
            for i in range(self.population_size):
                bot = self.population[i]
                for _ in range(5000):
                    observation, reward, terminated, truncated, info = self.env.step(bot.action)
                    action = bot.act(observation)
                    self.scores[i] += reward
                    if terminated or truncated:
                        break
            print(f"Generation {generation} finished with average score: {np.mean(self.scores)}")
            self.evolve()
        self.env.close()
    def crossover(self, bot1, bot2):
        bot = self.bot_class(self.env)
        for i in range(len(bot1.weights)):
            bot.weights[i] = bot1.weights[i] if random.random() > 0.5 else bot2.weights[i]
        return bot 
    def evolve(self):
        MUTATION_PROBABILITY = 0.1
        SURVIVOR_NUMBER = 20
        NEW_CROSSOVER_BOTS = 80

        sorted_indexes = np.argsort(self.scores)
        best_scores = [self.scores[i] for i in sorted_indexes]

        sorted_population = [self.population[i] for i in sorted_indexes]

        # select the 20 best bots 
        best_bots = sorted_population[:SURVIVOR_NUMBER]

        # create 80 new bots from the best 20
        # new_bots = [self.crossover(best_bots[i], best_bots[j]) for i in range(len(best_bots)) for j in range(i + 1, len(best_bots))]
        new_bots = []
        bot_count = 0
        for i in range(len(best_bots)):
            for j in range(i + 1, len(best_bots)):
                new_bots.append(self.crossover(best_bots[i], best_bots[j]))
                bot_count += 1
                if bot_count >= NEW_CROSSOVER_BOTS:
                    break
            if bot_count >= NEW_CROSSOVER_BOTS:
                break

        best_bots.extend(new_bots)
        # add the new bots to the population
        self.population = best_bots
        self.population_size = len(self.population)
        # reset the scores
        self.scores = [0 for _ in range(self.population_size)]

        # mutate the genes of the next generation
        for bot in self.population:
            for i in range(len(bot.weights)):
                if random.random() < MUTATION_PROBABILITY:
                    bot.weights[i] = bot.weights[i] + random.random()