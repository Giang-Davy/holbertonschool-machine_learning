#!/usr/bin/env python3
"""fonction"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2-GRADIENT"""
    m = Y.shape[1]
    weights_copy = weights.copy()

    for i in range(L, 0, -1):
        A_curr = cache["A" + str(i)]
        A_prev = cache["A" + str(i - 1)] if i > 1 else cache["A0"]

        if i == L:
            dZ = A_curr - Y
        else:
            dZ = dA * (1 - A_curr ** 2)

        dW = (np.dot(dZ, A_prev.T) / m) + (lambtha / m) * weights_copy[
            "W" + str(i)]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA = np.dot(weights_copy["W" + str(i)].T, dZ)

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
