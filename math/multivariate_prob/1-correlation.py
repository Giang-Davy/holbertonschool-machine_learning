#!/usr/bin/env python3
"""1-correlation.py"""


import numpy as np


def correlation(C):
    """correlation"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    
    ecart = np.sqrt(np.diag(C))
    outer = np.outer(ecart, ecart)
    correlation_matrix = C / outer
    return correlation_matrix
