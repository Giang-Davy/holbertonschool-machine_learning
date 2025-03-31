#!/usr/bin/env python3
"""0-initialize.py"""


import numpy as np


def initialize(X, k):
    """initialisation"""
    if not isinstance(X, np.ndarray ) or X.ndim != 2:
        return None
    mini = np.min(X, axis=0)
    maxi = np.max(X, axis=0)
    size = (k, X.shape[1])
    uni = np.random.uniform(mini, maxi, size)
    return uni
