#!/usr/bin/env python3
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    m = Y.shape[1]
    A = cache['A' + str(L)]
    dZ = A - Y
    for i in range(L, 0, -1):
        dW = (1 / m) * np.dot(dZ, cache['A' + str(i - 1)].T) + (lambtha / m) * weights['W' + str(i)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dZ = np.dot(weights['W' + str(i)].T, dZ) * (1 - np.power(cache['A' + str(i - 1)], 2))
