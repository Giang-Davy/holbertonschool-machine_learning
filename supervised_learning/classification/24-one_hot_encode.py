#!/usr/bin/env python3
"""fonction"""


import numpy as np


def one_hot_encode(Y, classes):
    """ff"""
    m = Y.shape[0]  # Nombre d'exemples
    if m == 0:
        return None
    if classes <= 0:
        return None

    one_hot_matrix = np.zeros((classes, m))

    # Remplissage de la matrice one-hot
    for i in range(m):
        if Y[i] >= classes or Y[i] < 0:
            return None
        one_hot_matrix[Y[i], i] = 1

    return one_hot_matrix
