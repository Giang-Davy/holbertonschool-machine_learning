#!/usr/bin/env python3
"""2-sarsa_lambtha.py"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """choix pour epsilon"""
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Utilisation de SARSA"""
    initial_epsilon = epsilon
    for i in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        eligibility_traces = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = reward + gamma * Q[
                next_state][next_action] - Q[state][action]
            eligibility_traces[state, action] += 1
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= lambtha * gamma
            if terminated or truncated:
                break
            state = next_state
            action = next_action
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))
    return Q
