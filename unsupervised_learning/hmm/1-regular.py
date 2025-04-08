#!/usr/bin/env python3
"""1-regular.py"""


import numpy as np


def regular(P):
    """Détermine si une matrice de transition est régulonnaire."""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    for _ in range(1, 101):
        temp = np.linalg.matrix_power(P, _)
        if np.all(temp > 0):
            break
    else:
        return None

    # Calcule la distribution stationnaire
    s = np.ones((1, P.shape[0])) / P.shape[0]
    for _ in range(10000):
        s_copy = s @ P
        if np.allclose(s_copy, s, atol=1e-8):
            return s_copy
        s = s_copy

    return None
