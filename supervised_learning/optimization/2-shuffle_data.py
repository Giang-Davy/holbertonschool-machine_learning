#!/usr/bin/env python3
"""fonction"""


import numpy as np


def shuffle_data(X, Y):
    """shuffle"""
    # Générer une permutation aléatoire des indices
    indices = np.random.permutation(X.shape[0])

    # Appliquer cette permutation aux matrices X et Y
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]

    return X_shuffle, Y_shuffle
