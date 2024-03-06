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
        self.weights = np.random.uniform(0, 1, size=(8, 4)) # 8 observations, 4 actions TODO : change this to be dynamic
        self.score = 0

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
    def __init__(self, env, bot_class: Bot, population_size=100, generations=100, mutation_probability=0.15, survivor_number=40, new_crossover_bots=60, starting_population=None):
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
        self.prev_bots = None

    def reset_population_score(self):
        for bot in self.population:
            bot.score = 0
    def run(self):
        print(f"Running genetic algorithm with {self.population_size} bots for {self.generations} generations.")
        generation_scores = []
        for generation in range(self.generations):
            self.prev_bots = self.population.copy()
            for i in range(len(self.population)): # for each bot calculate fitness (= total reward)
                bot = self.population[i]
                total_reward = self.play_bot(bot)
                bot.score = total_reward

            self.population = self.sort_bots_by_score(self.population)

            scores_of_generation = [bot.score for bot in self.population]
            mean_score_of_generation = np.mean(scores_of_generation)
            generation_scores.append(mean_score_of_generation)
            print(scores_of_generation)
            print(f"Generation {generation} finished with average score: {mean_score_of_generation}")

            if mean_score_of_generation >= 101:
                print("🎉 MEAN SCORE IS OVER 100 !!!!!!!!!!!! 🎉")
                self.write_json_to_file(f"genetic_algorithm_{time.time()}.json")
            if max(scores_of_generation) >= 100:
                print(f"Some bot has reached a good score of {max(scores_of_generation)}")

            mean_score_of_prev_generation = np.mean([bot.score for bot in self.prev_bots])
            if generation > 0 and mean_score_of_generation < mean_score_of_prev_generation:
                print(f"Regressing bad generation ...")
                self.population = self.prev_bots.copy()

            self.evolve()
                        
            self.reset_population_score()

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
        result_bot1 = Bot(bot1.env)
        result_bot2 = Bot(bot2.env)
        crossover_point = random.randint(0, len(bot1.weights))
        for i in range(len(bot1.weights)):
            if i > crossover_point:
                result_bot1.weights[i] = bot2.weights[i]
                result_bot2.weights[i] = bot1.weights[i]
            else:
                result_bot1.weights[i] = bot1.weights[i]
                result_bot2.weights[i] = bot2.weights[i]
        return result_bot1, result_bot2
    
    def random_crossover(self, bot1, bot2):
        result_bot1 = Bot(bot1.env)
        result_bot2 = Bot(bot2.env)
        for i in range(len(bot1.weights)):
            if random.random() > 0.5:
                result_bot1.weights[i] = bot2.weights[i]
                result_bot2.weights[i] = bot1.weights[i]
            else:
                result_bot1.weights[i] = bot1.weights[i]
                result_bot2.weights[i] = bot2.weights[i]
        return result_bot1, result_bot2

    # by default sort by highest to lowest reverse
    def sort_bots_by_score(self, bots, reverse=True):
        return sorted(bots, key=lambda bot: bot.score, reverse=reverse)
    
    # working
    # normalize negative values of fitness / score because they don't work with roulette select
    # https://stackoverflow.com/questions/44430194/roulette-wheel-selection-with-positive-and-negative-fitness-values-for-minimizat
    def normalize_scores(self, scores: list):
        abs_min_score = abs(min(scores))
        for i in range(len(scores)):
            scores[i] = scores[i] + abs_min_score
        return scores
            
    def calculate_roulette_probabilities(self, bots):
        scores = self.normalize_scores([bot.score for bot in bots])
        sum_of_fitness = sum(scores)
        previous_probability = 0.0
        probabilities = []
        for i in range(len(scores)):
            previous_probability = previous_probability + (scores[i] / sum_of_fitness)
            probabilities.append(previous_probability)
        for j in range(len(probabilities)):
            #print(f"bot : {scores[j]} has {probabilities[j]} % chance")
            pass
        return probabilities
    # think it works
    def roulette_select_n_bots(self, bots, n):
        bots = self.sort_bots_by_score(bots, False)
        probabilities = self.calculate_roulette_probabilities(bots)
        
        selected = []
        for j in range(n):
            random_number = random.random()
            for i in range(len(probabilities)):
                if random_number < probabilities[i]:
                    #print(f"selected bot with score : {bots[i].score} and probability : {probabilities[i]}")
                    selected.append(bots[i])
                    break
        return selected
    def evolve(self):
        new_population = []
        # selection, keep the best scoring bots of the previous and cur gen
        """
        if self.prev_bots is not None:
            if len(self.population) != len(self.prev_bots):
                print("Cant compare the previous and current generation since they are not same size")
            prev_bots_scores = [ bot.score for bot in self.prev_bots ]
            current_bots_scores = [ bot.score for bot in self.population ]

            new_bots_scores = []
            print(f"Selecting old ({np.mean(prev_bots_scores)}) vs current bots({np.mean(current_bots_scores)})")
            # selection of the best from prev gen x cur gen
            for i in range(len(prev_bots_scores)):
                if prev_bots_scores[i] > current_bots_scores[i]:
                    new_population.append(self.prev_bots[i])
                    new_bots_scores.append(self.prev_bots[i].score)
                else:
                    new_population.append(self.population[i])
                    new_bots_scores.append(self.population[i].score)
            if np.mean([bot.score for bot in self.population]) > np.mean(new_bots_scores): # mean score of the prev gen is better than the current gen
                print("mean score of the prev gen is better than the new selected gen (best of prev gen + best of cur gen)")
                return
            self.population = new_population

        self.population_size = len(self.population)
        """
        survivors = self.roulette_select_n_bots(self.population, self.survivor_number)
        elites = [ for bot in self.population if bot.score > 0 ]
        print(f"{len(survivors)} survived the roulette")
        print(f"Crossovering with population len {len(self.population)}")

        new_population = survivors.copy()
        print(f"Best {len(new_population)} choosen, crossovering")
        number_to_choose = self.survivor_number - (len(self.population) - len(new_population))
        for i in range(len(new_population)):
            for j in range(1, len(new_population)-1):
                (new_bot_1, new_bot_2) = self.random_crossover(new_population[i], new_population[j])
                if len(new_population) + 1 > number_to_choose:
                    break
                new_population.append(new_bot_1)
                if len(new_population) + 1 > number_to_choose:
                    break
                new_population.append(new_bot_2)

            if len(new_population) + 1 > number_to_choose:
                break
        self.population = new_population.copy()
        print(f"New population of {len(self.population)}")
        print("Mutating")
        for bot in self.population:
            bot.weights = self.mutate_matrix(bot.weights, self.mutation_probability)

    def mutate_matrix(self, matrix, mutation_rate, mutation_range=(0.25, 2), max_distance=0.3):
        #print(f"Mutating matrix with mutation rate {mutation_rate} and mutation range {mutation_range} and max distance {max_distance}")
        while True:
            mutated_matrix = np.copy(matrix)
            mutation_mask = np.random.rand(*matrix.shape) < mutation_rate
            mutation_values = np.random.uniform(mutation_range[0], mutation_range[1], size=matrix.shape)
            if random.random() > 0.5:
                mutated_matrix[mutation_mask] += mutation_values[mutation_mask]
            else:
                mutated_matrix[mutation_mask] -= mutation_values[mutation_mask]
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