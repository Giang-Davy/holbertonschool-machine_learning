#!/usr/bin/env python3
"""fonction"""


import numpy as np


class Neuron:
    """
    Classe qui présente les neuronnes
    """
    def __init__(self, nx):
        """
        Args: ff
        Returns: ff
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive")
        self._W = np.random.randn(1, nx)
        self._b = 0
        self._A = 0

    @property
    def W(self):
        """getter function"""
        return self._W

    @property
    def b(self):
        """getter function"""
        return self._b

    @property
    def A(self):
        """getter function"""
        return self._A

    def forward_prop(self, X):
        """
        Calcule la propagation avant du neurone

        Args:
            X: numpy.ndarray avec shape (nx, m) contenant les données d'entrée

        Returns:
            Activation (A): sortie activée du neurone
        """
        Z = np.dot(self._W, X) + self._b
        self._A = 1 / (1 + np.exp(-Z))
        return self._A

    def cost(self, Y, A):
        """ffland2"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """evaluer A et prédire"""
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """définir un neuron unique"""
        m = Y.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self._W -= alpha * dW
        self._b -= alpha * db
