#!/usr/bin/env python3
"""fontion"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """dropout_gradient_descent"""
    m = Y.shape[1]
    grads = {}
    dZL = cache['A' + str(L)] - Y
    grads['dW' + str(L)] = np.dot(dZL, cache['A' + str(L - 1)].T) / m
    grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m

    for i in range(L - 1, 0, -1):
        dA = np.dot(weights['W' + str(i + 1)].T, dZL)
        dA *= cache['D' + str(i)]
        dA /= keep_prob
        dZ = dA * (1 - np.power(cache['A' + str(i)], 2))
        grads['dW' + str(i)] = np.dot(dZ, cache['A' + str(i - 1)].T) / m
        grads['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m
        dZL = dZ

    for i in range(1, L + 1):
        weights['W' + str(i)] -= alpha * grads['dW' + str(i)]
        weights['b' + str(i)] -= alpha * grads['db' + str(i)]

    return weights
