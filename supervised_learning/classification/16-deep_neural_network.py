#!/usr/bin/env python3
"""fonction"""


import numpy as np


class DeepNeuralNetwork:
    """réseau neuronne profond"""
    def __init__(self, nx, layers):
        """initialisation"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Vérification de layers sans boucle explicite ou implicite
        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            prev = nx if i == 0 else layers[i - 1]
            self.weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
