#!/usr/bin/env python3
"""fonction"""

import numpy as np
import matplotlib.pyplot as plt

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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Entraîne le neurone"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations):
            # Propagation avant pour calculer A
            A1, A2 = self.forward_prop(X)
            # Descente de gradient pour mettre à jour les poids et le biais
            self.gradient_descent(X, Y, A1, A2, alpha)

            # Enregistrer le coût tous les 'step' itérations
            if i % step == 0 or i == iterations - 1:  # Inclure la dernière itération
                cost = self.cost(Y, self.__A2)
                costs.append(cost)
                if verbose:
                    iteration_display = i + 1 if i == iterations - 1 else i
                    print(f"Cost after {iteration_display} iterations: {cost}")

                
                

        # Graphique si demandé
        if graph:
            plt.plot(range(0, iterations + 1, step), costs, color="blue")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Retourne les prédictions et le coût final
        predictions, cost = self.evaluate(X, Y)
        return predictions, cost
