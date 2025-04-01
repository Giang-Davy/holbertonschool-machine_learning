#!/usr/bin/env python3
"""3-optimum.py"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin or kmax > n:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax - kmin + 1 < 2:
        return None, None

    results = []
    variances = []

    # Boucle unique pour calculer les résultats et les variances
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    # Calcul des différences de variance
    var_min = variances[0]
    d_vars = [var_min - var for var in variances]

    return results, d_vars
