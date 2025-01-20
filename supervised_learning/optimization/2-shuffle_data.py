#!/usr/bin/env python3
"""fonction"""


import numpy as np


def shuffle_data(X, Y):
    """shuffle"""
    X_shuffle = np.random.permutation(X)
    Y_shuffle = np.random.permutation(Y)

    return X_shuffle, Y_shuffle
