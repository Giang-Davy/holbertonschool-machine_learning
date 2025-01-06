#!/usr/bin/env python3
"""fonction"""


import numpy as np


class NeuralNetwork:
    """Réseau neuronne"""
    def __init__(self, nx, nodes):
        """ff"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self._W1 = np.random.randn(nodes, nx)
        self._b1 = np.zeros((nodes, 1))
        self._A1 = 0
        self._W2 = np.random.randn(1, nodes)
        self._b2 = 0
        self._A2 = 0

    @property
    def W1(self):
        """getter"""
        return self._W1

    @property
    def b1(self):
        """getter"""
        return self._b1

    @property
    def A1(self):
        """getter"""
        return self._A1

    @property
    def W2(self):
        """getter"""
        return self._W2

    @property
    def b2(self):
        """getter"""
        return self._b2

    @property
    def A2(self):
        """getter"""
        return self._A2

    def forward_prop(self, X):
        """propagation"""
        Z1 = np.dot(self._W1, X) + self._b1
        self._A1 = 1 / (1 + np.exp(-Z1))
        
        Z2 = np.dot(self._W2, self._A1) + self._b2
        self._A2 = 1 / (1 + np.exp(-Z2))
        
        return self._A1, self._A2
