#!/usr/bin/env python3
"""2-sarsa_lambtha.py"""


import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    nA = Q.shape[1]
    for i in range(episodes):
        state = env.reset()[0]
        e = np.zeros_like(Q)
        # Choix de l'action initiale
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(Q[state])
        for step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            # Choix de la prochaine action
            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(nA)
            else:
                next_action = np.argmax(Q[next_state])
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            e[state, action] += 1
            e *= gamma * lambtha
            Q += alpha * delta * e
            state = next_state
            action = next_action
        # Epsilon decay à chaque épisode
        if terminated or truncated:
                break
        epsilon = max(epsilon * (1 - epsilon_decay), min_epsilon)
    return Q
