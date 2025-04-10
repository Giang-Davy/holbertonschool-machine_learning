#!/usr/bin/env python3
"""Viterbi Algorithm"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """viterbi"""
    # Input validation
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if (not isinstance(Initial, np.ndarray) or
        Initial.ndim != 2 or
        Initial.shape[1] != 1):
        return None, None
    if (Emission.shape[0] != Transition.shape[0] or
        Transition.shape[0] != Transition.shape[1]):
        return None, None
    if Emission.shape[0] != Initial.shape[0]:
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    for i in range(N):
        F[i, 0] = Initial[i] * Emission[i, Observation[0]]
        backpointer[i, 0] = 0

    for t in range(1, T):
        for i in range(N):
            trans_probs = F[:, t-1] * Transition[:, i] * Emission[i, Observation[t]]
            F[i, t] = np.max(trans_probs)
            backpointer[i, t] = np.argmax(trans_probs)

    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(F[:, T-1])

    for t in range(T-2, -1, -1):
        path[t] = backpointer[path[t+1], t+1]

    P = np.max(F[:, T-1])

    return path, P
