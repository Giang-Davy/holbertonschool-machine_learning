#!/usr/bin/env python3
"""8-EM.py"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """maximization attente"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    prev_like = None
    pi, m, S = initialize(X, k)
    for i in range(iterations):
        g, like = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if prev_like is not None and abs(like - prev_like) <= tol:
            break
        prev_like = like
    return pi, m, S, g, like
