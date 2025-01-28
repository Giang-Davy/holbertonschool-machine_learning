#!/usr/bin/env python3
"""fonction"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2 gradient"""
    m = Y.shape[1]

    for l in range(L, 0, -1):
        A = cache[f'A{l}']
        if l == L:
            dz = A - Y
        else:
            dz = np.multiply(np.dot(weights[f'W{l+1}'].T, dz), (1 - np.power(A, 2)))

        dw = np.dot(dz, cache[f'A{l-1}'].T) / m + (lambtha / m) * weights[f'W{l}']
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights[f'W{l}'] -= alpha * dw
        weights[f'b{l}'] -= alpha * db
