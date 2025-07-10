#!/usr/bin/env python3
"""1-td_lambtha.py"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """algorithm avec td_lambda"""
    for i in range(episodes):
        e = np.zeros_like(V)
        state = env.reset()[0]  # régler le problème de tuple

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            # Incrémente la trace d'éligibilité pour l'état courant
            e[state] += 1

            # Met à jour tous les états selon leur trace d'éligibilité
            V += alpha * delta * e

            # Décroît toutes les traces d'éligibilité
            e *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break
    return V
