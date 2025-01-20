#!/usr/bin/env python3
"""fonction"""

import numpy as np


def normalization_constants(X):
    """calcule de la moyenne et de l'ecart type"""
    m = X.shape[0]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
