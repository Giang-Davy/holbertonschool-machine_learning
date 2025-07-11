#!/usr/bin/env python3
"""2-sarsa_lambtha.py"""


import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Utilisation de SARSA"""
    initial_epsilon = epsilon
    nA = Q.shape[1]
    for i in range(episodes):
        state = env.reset()[0]
        e = np.zeros_like(Q)
        # Choix de l'action initiale
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(Q[state])
        eligibility_traces = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            # Choix de la prochaine action
            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(nA)
            else:
                next_action = np.argmax(Q[next_state])
            delta = reward + gamma * Q[
                next_state][next_action] - Q[state][action]
            eligibility_traces[state, action] += 1
            eligibility_traces *= lambtha * gamma
            Q += alpha * delta * eligibility_traces
            state = next_state
            action = next_action
            if terminated or truncated:
                break
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))
    return Q
