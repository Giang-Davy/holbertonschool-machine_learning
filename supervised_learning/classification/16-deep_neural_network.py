#!/usr/bin/env python3
"""fonction"""


import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer_idx in range(self.L):
            layer_input = nx if layer_idx == 0 else layers[layer_idx - 1]
            self.weights[f"W{layer_idx + 1}"] = (
                np.random.randn(layers[layer_idx], layer_input) * np.sqrt(2 / layer_input)
            )
            self.weights[f"b{layer_idx + 1}"] = np.zeros((layers[layer_idx], 1))
