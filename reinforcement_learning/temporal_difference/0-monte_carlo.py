#!/usr/bin/env python3
"""0-monte_carlo.py"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Monte Carlo Algorithm"""

    for i in range(episodes):
        episode = []
        state = env.reset()[0]  # régler le problème de tuple
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break
            state = next_state
        G = 0
        episode = np.array(episode, dtype=int)
        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in episode[:i, 0]:
                # Update the value function V(s)
                V[state] = V[state] + alpha * (G - V[state])
    return V
