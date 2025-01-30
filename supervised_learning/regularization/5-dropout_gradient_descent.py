#!/usr/bin/env python3
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network with Dropout regularization using gradient descent.
    
    Y : numpy.ndarray, shape (classes, m) — true labels
    weights : dictionary containing weights and biases
    cache : dictionary containing outputs and dropout masks for each layer
    alpha : learning rate
    keep_prob : probability that a neuron will be kept
    L : number of layers in the network
    """
    m = Y.shape[1]  # number of examples
    
    # Initialize gradients
    grads = {}

    # Backward propagation for the last layer (Softmax)
    dZL = cache['A' + str(L)] - Y  # Calculate the error of the last layer
    grads['dW' + str(L)] = np.dot(dZL, cache['A' + str(L - 1)].T) / m
    grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m

    # Backward propagation for hidden layers
    for i in range(L - 1, 0, -1):
        dA = np.dot(weights['W' + str(i + 1)].T, dZL)
        dA *= cache['D' + str(i)]  # Apply dropout mask
        dA /= keep_prob  # Scale the dropout mask
        dZ = dA * (1 - np.power(cache['A' + str(i)], 2))  # Derivative of tanh

        grads['dW' + str(i)] = np.dot(dZ, cache['A' + str(i - 1)].T) / m
        grads['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m

        dZL = dZ  # Update the error for the next iteration

    # Update weights and biases using gradient descent
    for i in range(1, L + 1):
        weights['W' + str(i)] -= alpha * grads['dW' + str(i)]
        weights['b' + str(i)] -= alpha * grads['db' + str(i)]

    return weights

