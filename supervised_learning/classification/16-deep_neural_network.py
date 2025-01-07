#!/usr/bin/env python3
"""fonction"""


import numpy as np


class DeepNeuralNetwork:
    """réseau neurronne profond"""
    def __init__(self, nx, layers):
        """initialisation"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(layer, int) or layer <= 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            layer_in = nx if i == 0 else layers[i - 1]
            self.weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], layer_in) * np.sqrt(2 / layer_in)
            )
            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
