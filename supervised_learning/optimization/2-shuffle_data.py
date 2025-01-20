#!/usr/bin/env python3
"""fonction"""


import numpy as np


def shuffle_data(X, Y):
    """shuffle"""
    np.random.permutation(X)
    np.random.permutation(Y)
    
    return X, Y
