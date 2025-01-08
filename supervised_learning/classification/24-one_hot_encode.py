#!/usr/bin/env python3
"""fonction"""


import numpy as np


def one_hot_encode(Y, classes):
    """method"""
    if not isinstance(Y, np.ndarray) or Y.ndim != 1 or not isinstance(
             classes, int) or classes <= 0:
        return None
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    for i in range(m):
        if Y[i] >= classes or Y[i] < 0:
            return None
        one_hot[Y[i], i] = 1
    return one_hot
