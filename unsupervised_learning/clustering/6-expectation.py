#!/usr/bin/env python3
"""6-expectation"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if m.shape[0] != pi.shape[0] or m.shape[1] != X.shape[1]:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3 or S.shape[2] != X.shape[1]:
        return None, None
    if S.shape[0] != pi.shape[0] or S.shape[1] != X.shape[1]:
        return None, None

    k, n = pi.shape[0], X.shape[0]
    g = np.zeros((k, n))

    for i in range(k):  
        g[i] = pi[i] * pdf(X, m[i], S[i])

    total_prob = np.sum(g, axis=0)
    g /= total_prob

    l = np.sum(np.log(total_prob))

    return g, l
