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
        if min(layers) < 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            prev = nx if i == 0 else layers[i - 1]
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
