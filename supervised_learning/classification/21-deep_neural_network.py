#!/usr/bin/env python3
"""fonction"""


import numpy as np


np.random.seed(42)  # Fixer la graine pour des résultats reproductibles


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

    @property
    def L(self):
        """getter"""
        return self.__L

    @property
    def cache(self):
        """getter"""
        return self.__cache

    @property
    def weights(self):
        """getter"""
        return self.__weights

    def forward_prop(self, X):
        """Propagation avant du réseau de neurones."""
        self.__cache['A0'] = X

        A = X
        for i in range(1, self.__L + 1):
            Z = np.dot(self.__weights[f'W{i}'], A) + self.__weights[f'b{i}']
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A

        return A, self.__cache

    def cost(self, Y, A):
        """fonnction coût"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Évalue les prédictions du réseau de neurones et calcule le coût"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Effectue une passe de descente de gradient"""
        m = Y.shape[1]
        L = self.__L

        # Calcul de dZ pour la dernière couche
        dZ = cache[f'A{L}'] - Y

        # Rétropropagation à travers toutes les couches
        for index_couche in range(L, 0, -1):
            # Récupération des activations de la couche précédente
            A_prev = cache[f'A{index_couche-1}']

            # Calcul des gradients
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if index_couche > 1:
                # Calcul de dZ pour la couche précédente
                W = self.__weights[f'W{index_couche}']
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            # Mise à jour des poids et biais
            self.__weights[f'W{index_couche}'] -= alpha * dW
            self.__weights[f'b{index_couche}'] -= alpha * db
