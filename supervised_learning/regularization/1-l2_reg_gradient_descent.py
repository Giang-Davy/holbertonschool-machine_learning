#!/usr/bin/env python3
"""fonction"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]

    A_L = cache["A" + str(L)]
    dZ_L = A_L - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)] if i > 1 else cache["A0"]
        W = weights["W" + str(i)]

        dW = np.dot(dZ_L, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ_L, axis=1, keepdims=True) / m

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
        if i > 1:
            dZ_L = np.dot(W.T, dZ_L) * (
                1 - np.power(cache["A" + str(i - 1)], 2))

    return weights
