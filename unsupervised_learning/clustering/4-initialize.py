#!/usr/bin/env python3
"""4-initialize.py"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initialisation"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None
    pi = np.full(k, 1/k)
    m, _ = kmeans(X, k)
    # c'est nouveau ça
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
