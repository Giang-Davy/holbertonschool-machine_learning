#!/usr/bin/env python3
"""fonction"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Droupout-avant"""
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax activation function for the last layer
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            # Tanh activation function for hidden layers
            A = np.tanh(Z)

            # Dropout mask (convert False/True to 0/1)
            D = (np.random.rand(
                A.shape[0], A.shape[1]) < keep_prob).astype(int)
            A = A * D / keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
