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
    """Algorithme des K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        C_old = C.copy()
        # 1. Calculer les distances entre les points et les centroids
        distances = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))

        # 2. Assigner chaque point au cluster le plus proche
        clss = np.argmin(distances, axis=0)

        for i in range(k):
            # Mask: points present in cluster
            cluster_mask = X[clss == i]
            if len(cluster_mask) == 0:
                C[i] = initialize(X, 1)
            else:
                C[i] = np.mean(X[clss == i], axis=0)

        # Recalculate distances and reassign clusters
        dists = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        # 5. Vérifier la convergence
        if np.all(C == C_old):
            break

    return C, clss


def variance(X, C):
    """variance-total"""
    dists = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
    clss = np.argmin(dists, axis=0)
    variance = np.sum(np.linalg.norm(X - C[clss], axis=1) ** 2)
    return variance
