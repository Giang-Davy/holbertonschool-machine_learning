#!/usr/bin/env python3
"""0-monte_carlo.py"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Monte Carlo Algorithm"""

    for i in range (episodes):
        episode = []
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(
                action)
            if isinstance(state, tuple):
                state = state[0]
            done = terminated or truncated
            episode.append((state, reward))

            if done:
                break
            state = next_state
        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])
    return V
