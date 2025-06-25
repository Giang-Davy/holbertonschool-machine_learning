#!/usr/bin/env python3
"""2-epsilon_greedy.py"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """espil"""

    extract = Q[state]
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
        return action
    else:
        best_action = np.argmax(extract)
        return best_action
