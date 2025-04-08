#!/usr/bin/env python3
"""1-regular.py"""


import numpy as np


def regular(P):
    """reguleur"""
    if not isinstance(P, np.ndarray):
        return None
    temp = P.copy()
    for _ in range(1, 101):
        temp = np.linalg.matrix_power(P, _)
        if np.all(temp > 0):
            break
    else:
        return None
    s = np.ones((1, P.shape[0])) / P.shape[0]
    for _ in range(10000):
        s_copy = s @ P
        if np.allclose(s_copy, s):
            return s
        s = s_copy

    return None
