#!/usr/bin/env python3
"""2-absorbing.py"""


import numpy as np


def absorbing(P):
    """absorber"""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n, m = P.shape
    if n != m:
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False

    absorbing_states = []
    for i in range(n):
        if P[i, i] == 1:
            absorbing_states.append(i)

    if len(absorbing_states) == 0:
        return False

    reachable = np.linalg.matrix_power(P, n)
    for i in range(n):
        if not any(reachable[i, j] > 0 for j in absorbing_states):
            return False

    return True
