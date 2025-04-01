#!/usr/bin/env python3
"""3-optimum.py"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimize"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape
    if not isinstance(kmin, int) or not isinstance(kmax, int) or kmin <= 0 or kmax < kmin or kmax > n:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    var_min = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var_min is None:  # Première valeur comme référence
            var_min = var
            d_vars.append(0)  # La première différence est 0
        else:
            delta_var = var_min - var  # Calculer la différence
            d_vars.append(delta_var if delta_var > 0 else 0)  # Ajouter uniquement des valeurs positives

    return results, d_vars
