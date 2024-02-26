import gymnasium as gym
import numpy as np

# Créer l'environnement Lunar Lander
env = gym.make("LunarLander-v2")

# Discrétisation de l'espace d'observation
n_bins = 20
state_bins = [np.linspace(l, h, n_bins) for l, h in zip(env.observation_space.low, env.observation_space.high)]

# Initialiser la table Q avec des valeurs aléatoires
Q = np.zeros(tuple(n_bins) + (env.action_space.n,))

# Paramètres de l'algorithme Q-learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.99  # Facteur de remise
epsilon = 0.1  # Valeur de l'epsilon pour la politique epsilon-greedy

# Fonction epsilon-greedy pour choisir une action
def epsilon_greedy_policy(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        indices = tuple(np.digitize(s, bins) - 1 for s, bins in zip(state, state_bins))
        return np.argmax(Q[indices])


# Boucle d'apprentissage
for episode in range(1, 1001):  # Vous pouvez ajuster le nombre d'épisodes selon vos besoins
    state = env.reset()
    total_reward = 0

    while True:
        action = epsilon_greedy_policy(state)

        # Effectuer l'action et observer le nouvel état et la récompense
        next_state, reward, done, _ = env.step(action)

        # Mettre à jour la valeur Q de l'état-action actuel
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))

        state = next_state
        total_reward += reward

        if done:
            break

    # Afficher la récompense totale de l'épisode
    if episode % 100 == 0:
        print(f"Épisode {episode}, Récompense totale : {total_reward}")

# Utiliser la politique apprise pour jouer une partie
state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state, :])
    state, _, done, _ = env.step(action)

# Fermer l'environnement
env.close()
