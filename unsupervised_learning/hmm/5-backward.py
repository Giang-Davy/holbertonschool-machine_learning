#!/usr/bin/env python3
"""
5-backward.py
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """backward"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] * Transition[i, :] * Emission[
                    :, Observation[t+1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
