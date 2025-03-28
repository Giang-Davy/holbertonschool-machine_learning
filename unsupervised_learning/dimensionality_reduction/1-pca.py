#!/usr/bin/env python3


import numpy as np


def pca(X, ndim):
    """calculer pca 2"""
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = Vt[:ndim].T
    T = np.dot(X_centered, W)

    return T
