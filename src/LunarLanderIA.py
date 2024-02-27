import json
import os
import random
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class LunarLanderPlayer:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.actions = list(range(self.env.action_space.n))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_end = 0.01
        self.previous_state = None
        self.previous_action = None

        self.qvalues_file = "qvalues.json"
        if os.path.exists(self.qvalues_file):
            with open(self.qvalues_file, 'r') as file:
                self.qvalues = json.load(file)
        else:
            self.qvalues = {}

    def discretize_state(self, state):
        return tuple(map(float, state[0]))

    def decide(self, state):
        state = self.discretize_state(state)

        # Adaptive exploration rate decay
        eps = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        print(f"State: {state}, Epsilon: {eps}")
        print(f"QValues: {self.qvalues}")
        print(f"Actions: {self.actions}")

        # Softmax exploration
        state_key = str(state)
        
        if state_key not in self.qvalues:
            # Handle the case where the key is not present in the dictionary
            print(f"State key {state_key} not found in QValues. Using random action.")
            action = random.choice(self.actions)
        else:
            action_values = [self.qvalues[state_key][a]['score'] for a in self.actions]
            probabilities = self.softmax(action_values, eps)
            action = np.random.choice(self.actions, p=probabilities)

        if self.previous_state is not None:
            self.update_qvalues(self.previous_state, self.previous_action, 0, state, action)

        self.previous_state = state
        self.previous_action = action
        return action

    def softmax(self, action_values, epsilon):
        exp_values = np.exp(np.array(action_values) / epsilon)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def search_best_action(self, state, eps):
        state_key = str(state)
        if state_key not in self.qvalues:
            # Initialize scores for all actions
            self.qvalues[state_key] = {action: {'score': 0, 'count': 0} for action in self.actions}

        action_values = [self.qvalues[state_key][a]['score'] for a in self.actions]
        probabilities = self.softmax(action_values, eps)

        action = np.random.choice(self.actions, p=probabilities)
        return action

    def update_qvalues(self, state_t1, action_t1, reward, state_t2, action_t2):
        state_t1_key = str(state_t1)
        state_t2_key = str(state_t2) if state_t2 is not None else None

        if state_t1_key not in self.qvalues:
            self.qvalues[state_t1_key] = {action: {'score': 0, 'count': 0} for action in self.actions}

        if state_t2_key and state_t2_key not in self.qvalues:
            self.qvalues[state_t2_key] = {action: {'score': 0, 'count': 0} for action in self.actions}

        # Clip the reward
        reward = np.clip(reward, -1, 1)

        old_value = self.qvalues[state_t1_key][action_t1]['score']
        future_value = self.qvalues[state_t2_key][action_t2]['score'] if state_t2_key else 0

        self.qvalues[state_t1_key][action_t1]['score'] = (
        1 - self.alpha
            ) * old_value + self.alpha * (reward + self.gamma * future_value)
        self.qvalues[state_t1_key][action_t1]['count'] += 1

    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon

        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()
            print(f"State {state}")
            score = 0

            for t in range(max_t):
                action = self.decide(state)
                next_state, reward, done, _, _ = self.env.step(action)

                eps = max(eps_end, eps_decay * eps)
                best_action = self.search_best_action(next_state, eps)

                self.update_qvalues(state, action, reward, next_state, best_action)
                state = next_state
                score += reward

                if done:
                    break 

            scores_window.append(score)
            scores.append(score)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break

        # Plot the learning progress
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

if __name__ == "__main__":
    player = LunarLanderPlayer()
    player.train()