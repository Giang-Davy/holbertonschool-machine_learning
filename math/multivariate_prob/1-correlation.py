#!/usr/bin/env python3
"""1-correlation.py"""


import numpy as np


def correlation(C):
    """correlation"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if not len(C.shape) == 2:
        raise ValueError("C must be a 2D square matrix")
    corr = np.corrcoef(C)
    return corr
