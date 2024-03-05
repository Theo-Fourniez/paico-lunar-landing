# a genetic algorithm bot class 
import json
import random
import time
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt

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
    
    def to_json(self):
        return {
            "weights": self.weights.tolist()
        }
    
    def read_from_json(self, path):
        with open(path, "r") as file:
            obj = json.loads(file.read())
            self.weights = np.array(obj["weights"])

    def write_to_json(self, path):
        # Write the object's content to a JSON file in a folder
        folder_path = "out" 
        file_name = f"{path}.json"  
        file_path = os.path.join(folder_path, file_name)

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        with open(file_path, "w") as file:
            file.write(json.dumps(self.to_json()))
            
            
# a genetic algorithm class
class GeneticAlgorithm:
    def __init__(self, env, bot_class: Bot, population_size=100, generations=1000, mutation_probability=0.1, survivor_number=30, new_crossover_bots=70, starting_population=None):
        if type(starting_population) is not list and starting_population is not None:
            raise ValueError("starting_population should be a list of Bot objects or None")
        self.env = env
        self.bot_class = bot_class
        self.population_size = population_size if starting_population is None else len(starting_population)
        self.generations = generations
        self.population = [bot_class(env) for _ in range(population_size)] if starting_population is None else starting_population
        self.scores = np.zeros(self.population_size)
        self.mutation_probability = mutation_probability
        self.survivor_number = survivor_number
        self.new_crossover_bots = new_crossover_bots
        self.prev_mean_score = 0

    def run(self):
        print(f"Running genetic algorithm with {self.population_size} bots for {self.generations} generations.")
        generation_scores = []

        for generation in range(self.generations):
            for i in range(self.population_size):
                bot = self.population[i]
                total_reward = self.play_bot(bot)
                self.scores[i] += total_reward

            mean_score_of_generation = np.mean(self.scores)
            generation_scores.append(mean_score_of_generation)
            print(f"Generation {generation} finished with average score: {mean_score_of_generation}")

            if mean_score_of_generation >= 101:
                print("🎉 MEAN SCORE IS OVER 100 !!!!!!!!!!!! 🎉")
                self.write_json_to_file(f"genetic_algorithm_{time.time()}.json")
            if self.scores.max() >= 100:
                print(f"Some bot has reached a good score of {self.scores.max()}")
                max_index = np.argmax(self.scores)
                best_bot = self.population[max_index]
                best_bot.write_to_json(f"best_bot_{time.time()}.json")
            self.evolve()

        # Plotting
        self.plot_generation_scores(generation_scores)
        self.env.close()

    def play_bot(self, bot):
        total_reward = 0
        for _ in range(5000):
            observation, reward, terminated, truncated, info = bot.step()
            bot.act(observation)
            total_reward += reward
            if terminated or truncated:
                self.env.reset()
                break
        return total_reward
    
    # the crossover should return two bots
    def one_point_crossover(self, bot1, bot2):
        result_bot1 = bot1
        result_bot2 = bot2
        crossover_point = random.randint(0, len(bot1.weights))
        for i in range(len(bot1.weights)):
            if i > crossover_point:
                result_bot1.weights[i] = bot2.weights[i]
                result_bot2.weights[i] = bot1.weights[i]
        return result_bot1, result_bot2
    
    def random_crossover(self, bot1, bot2):
        result_bot1 = bot1
        result_bot2 = bot2
        for i in range(len(bot1.weights)):
            if random.random() > 0.5:
                result_bot1.weights[i] = bot2.weights[i]
                result_bot2.weights[i] = bot1.weights[i]
        return result_bot1, result_bot2

    def evolve(self):
        sorted_indexes = np.argsort(self.scores)

        sorted_population = [self.population[i] for i in sorted_indexes]

        # select the 20 best bots 
        best_bots = sorted_population[:self.survivor_number]

        # create 80 new bots from the best 20
        # new_bots = [self.crossover(best_bots[i], best_bots[j]) for i in range(len(best_bots)) for j in range(i + 1, len(best_bots))]
        new_bots = []
        bot_count = 0
        for i in range(len(best_bots)):
            for j in range(i + 1, len(best_bots)):
                new_bots.extend(self.random_crossover(best_bots[i], best_bots[j]))
                bot_count += 2
                if bot_count >= self.new_crossover_bots:
                    break
            if bot_count >= self.new_crossover_bots:
                break

        best_bots.extend(new_bots)
        # add the new bots to the population
        self.population = best_bots
        self.population_size = len(self.population)
        # reset the scores
        self.scores = np.zeros(self.population_size)

        for bot in self.population:
            bot.weights = self.mutate_matrix(bot.weights, self.mutation_probability)

    def mutate_matrix(self, matrix, mutation_rate, mutation_range=(0.15, 1.5), max_distance=0.5):
        while True:
            mutated_matrix = np.copy(matrix)
            mutation_mask = np.random.rand(*matrix.shape) < mutation_rate
            mutation_values = np.random.uniform(mutation_range[0], mutation_range[1], size=matrix.shape)
            mutated_matrix[mutation_mask] += mutation_values[mutation_mask]
            distance = np.linalg.norm(mutated_matrix - matrix)
            if distance < max_distance:
                return mutated_matrix

    def to_json(self):
        obj = {
            "population_size": self.population_size,
            "generations": self.generations,
            "population": [bot.weights.tolist() for bot in self.population],
            "avg_score": np.mean(self.scores),
            "mutation_probability": self.mutation_probability,
            "survivor_number": self.survivor_number,
            "new_crossover_bots": self.new_crossover_bots
        }
        return obj
    
    def read_from_json(self, path):
        with open(path, "r") as file:
            obj = json.loads(file.read())
            self.population_size = obj["population_size"]
            self.generations = obj["generations"]
            self.population = [Bot(self.env) for weights in obj["population"]]
            self.scores = [0 for _ in range(self.population_size)]
            self.mutation_probability = obj["mutation_probability"]
            self.survivor_number = obj["survivor_number"]
            self.new_crossover_bots = obj["new_crossover_bots"]
    
    def write_json_to_file(self, path):
        with open(path, "w") as file:
            file.write(json.dumps(self.to_json()))
            
    def plot_generation_scores(self, generation_scores):
        generations = range(self.generations)
        plt.plot(generations, generation_scores, label='Average Score')
        plt.xlabel('Generation')
        plt.ylabel('Average Score')
        plt.title('Genetic Algorithm Performance')
        plt.legend()
        plt.show()