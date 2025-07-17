#!/usr/bin/env python3
"""train.py"""


import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """entrainement"""
    score_list = []
    weight = np.random.randn(env.observation_space.shape[0], env.action_space.n) * 0.01
    for episode in range(nb_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        description = []
        done = False
        while done == False:
            state = np.array(state)
            action, gradient = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, info = env.step(
                action)
            description.append((reward, gradient))
            state = next_state
            done = terminated or truncated
        score = sum(r for r, _ in description)
        score_list.append(score)
        print("Episode: {} Score: {}".format(episode, score))
        G = 0
        gradients_sum = np.zeros_like(weight)
        for reward, gradient in reversed(description):
            if not isinstance(gradient, np.ndarray):
                gradient = np.array(gradient)
            G = reward + gamma * G
            gradients_sum += gradient * G
        weight += alpha * gradients_sum
        
    return score_list
