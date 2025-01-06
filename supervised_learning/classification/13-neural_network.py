#!/usr/bin/env python3
"""fonction"""

import numpy as np


np.random.seed(42)


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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter"""
        return self.__W1

    @property
    def b1(self):
        """getter"""
        return self.__b1

    @property
    def A1(self):
        """getter"""
        return self.__A1

    @property
    def W2(self):
        """getter"""
        return self.__W2

    @property
    def b2(self):
        """getter"""
        return self.__b2

    @property
    def A2(self):
        """getter"""
        return self.__A2

    def forward_prop(self, X):
        """propagation"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """fonnction coût"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """evaluer A et prédire"""
        self.forward_prop(X)
        predictions = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Descente de gradient pour mettre à jour W et b"""
        m = Y.shape[1]

        # Calcul des dérivées pour la couche de sortie
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # Calcul des dérivées pour la première couche
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Mise à jour des paramètres de la deuxième couche
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        # Mise à jour des paramètres de la première couche
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
