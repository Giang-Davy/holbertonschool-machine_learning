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
        # 1. Calculer les distances entre les points et les centroids
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)

        # 2. Assigner chaque point au cluster le plus proche
        clss = np.argmin(distances, axis=0)

        # 3. Sauvegarder les anciens centroids
        C_old = C.copy()

        # 4. Mettre à jour les centroids
        for i in range(k):
            if np.any(clss == i):
                C[i] = np.mean(X[clss == i], axis=0)
            else:
                C[i] = np.random.uniform(
                    np.min(X, axis=0), np.max(X, axis=0), size=(1, d))

        # 5. Vérifier la convergence
        if np.all(C == C_old):
            break

    return C, clss
