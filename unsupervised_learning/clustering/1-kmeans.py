#!/usr/bin/env python3
"""1-kmeans.py"""


import numpy as np


def initialize(X, k):
    """initialisation"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None
    mini = np.min(X, axis=0)
    maxi = np.max(X, axis=0)
    size = (k, X.shape[1])
    uni = np.random.uniform(mini, maxi, size)
    return uni


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(2):  # Limiter à deux boucles
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i) else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape[1]) for i in range(k)])

        # Check for convergence
        if np.all(C == new_C):
            break
        C = new_C

    return C, clss
