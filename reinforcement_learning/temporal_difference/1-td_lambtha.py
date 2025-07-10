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

            if terminated or truncated:
                break
            delta = reward + gamma * V[next_state] - V[state]
            
            for s in range(len(V)):
                if s == state:
                    e[s] = gamma * lambtha * e[s] + 1
                else:
                    e[s] = gamma * lambtha * e[s]
                V[s] = V[s] + alpha * delta * e[s]
            state = next_state
    return V
