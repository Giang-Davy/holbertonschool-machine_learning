#!/usr/bin/env python3
"""0-pca.py"""
import numpy as np


def pca(X, var=0.95):
    """
    pca fonction
    """
    U, S, V = np.linalg.svd(X, full_matrices=False)

    var_ratio = np.cumsum(S**2) / np.sum(S**2)

    nb_comp = np.argmax(var_ratio >= var) + 1

    W = V[:nb_comp + 1].T

    return W
