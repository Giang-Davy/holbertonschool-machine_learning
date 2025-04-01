#!/usr/bin/env python3
"""2-variance.py"""


import numpy as np


def variance(X, C):
    """variance-total"""
    dists = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
    clss = np.argmin(dists, axis=0)
    variance = np.sum(np.linalg.norm(X - C[clss], axis=1) ** 2)
    return variance
