#!/usr/bin/env python3
"""1-regular.py"""


import numpy as np


def regular(P):
    """Determines the steady-state probabilities of a regular Markov chain."""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Check if the matrix is regular
    power = np.linalg.matrix_power(P, 100)
    if not np.all(power > 0):
        return None

    # Solve for the steady-state probabilities
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    steady_state = eigenvectors[:, np.isclose(eigenvalues, 1)]
    steady_state = steady_state / steady_state.sum()
    return steady_state.real.T
