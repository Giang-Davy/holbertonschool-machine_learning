#!/usr/bin/env python3
"""2-sarsa_lambtha.py"""


import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    for i in range(episodes):
        state = env.reset()[0]
        e = np.zeros_like(Q)
        for step in range(max_steps):
            extract = Q[state]
            p = np.random.uniform(0, 1)
            if p < epsilon:
                action = np.random.randint(Q.shape[1])
            else:
                action = np.argmax(extract)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                    break
            p_next = np.random.uniform(0, 1)
            next_extract = Q[next_state]
            if p_next < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(next_extract)
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            e[state, action] += 1
            Q += alpha * delta * e
            e *= gamma * lambtha
            state = next_state
            action = next_action
            epsilon = max(epsilon*(1-epsilon_decay), min_epsilon)
            return Q
