#!/usr/bin/env python3
"""3-q_learning.py"""


import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """"entrainement q learning"""

    total_reward = []

    for episode in range(episodes):
        # Réinitialisation de l’environnement
        state = env.reset()
        # Garantire la compatibilité avec les versions de gymnasium
        if isinstance(state, tuple):
            state = state[0]
        done = False
        rewards = 0
        for step in range(max_steps):
            # Exectuter une action
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(
                action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            done = terminated or truncated
            # Récompense négative si il tombe dans un trou
            if done and reward == 0:
                reward = -1
            # Mettre à jour le Q-table
            Q[state, action] = Q[state][action] + alpha * (
                reward + gamma * np.max(
                    Q[next_state])-Q[state][action])
            # Rassembler toute les récompenses
            rewards += reward
            if done:
                break
            state = next_state  # ligne pour mettre à jour l'état
            # Exploration
        epsilon = max(epsilon*(1-epsilon_decay), min_epsilon)
        # Sauvegarde des récompenses
        total_reward.append(rewards)
    return Q, total_reward
