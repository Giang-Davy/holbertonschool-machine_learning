#!/usr/bin/env python3
"""5-pdf.py"""


import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if m.shape[0] != X.shape[1]:
        return None
    if not isinstance(S, np.ndarray) or S.shape[0] != X.shape[1]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    det_cov = np.linalg.det(S)
    inv_cov = np.linalg.inv(S)
    X_centered = X - m
    coef = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * det_cov)
    # Correct PDF formula
    exponent = -0.5 * np.sum(X_centered @ inv_cov * X_centered, axis=1)

    P = coef * np.exp(exponent)
    return P
