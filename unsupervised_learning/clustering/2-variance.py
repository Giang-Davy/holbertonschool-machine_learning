#!/usr/bin/env python3
"""2-variance.py"""


import numpy as np


def variance(X, C):
    """variance-total"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    dists = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
    clss = np.argmin(dists, axis=0)
    variance = np.sum(np.linalg.norm(X - C[clss], axis=1) ** 2)
    return variance
