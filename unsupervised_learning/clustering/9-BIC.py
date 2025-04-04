#!/usr/bin/env python3
"""9-BIC.py"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """BIC"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin or kmax > n:
        return None, None, None, None
    if kmax - kmin + 1 < 2:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    likehood = []  # Liste pour stocker les log-vraisemblances
    b = []         # Liste pour stocker les valeurs BIC
    best_bic = float('inf')  # Initialisation du meilleur BIC à l'infini
    best_k = None
    best_result = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, like = expectation_maximization(
            X, k, iterations, tol, verbose)

        # Ajouter la log-vraisemblance à la liste
        likehood.append(like)

        p = k * (d + (d * (d + 1)) / 2 + 1)

        # Calcul du BIC
        BIC_value = p * np.log(n) - 2 * like

        # Ajouter la valeur BIC à la liste
        b.append(BIC_value)

        # Vérifier si ce BIC est le meilleur
        if k == kmin or BIC_value < best_bic:
            # Update the return values
            best_bic = BIC_value
            best_results = (pi, m, S)
            best_k = k

    likelihoods = np.array(likelihoods)
    b = np.array(b)

    return best_k, best_results, likelihoods, b
