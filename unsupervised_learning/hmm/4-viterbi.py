#!/usr/bin/env python3
"""Viterbi Algorithm"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """viterbi"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer[:, 0] = 0

    for t in range(1, T):
        for i in range(N):
            trans_probs = F[
                :, t-1] * Transition[:, i] * Emission[i, Observation[t]]
            F[i, t] = np.max(trans_probs)
            backpointer[i, t] = np.argmax(trans_probs)

    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(F[:, T-1])

    for t in range(T-2, -1, -1):
        path[t] = backpointer[path[t+1], t+1]

    P = np.max(F[:, T-1])

    return path, P
