#!/usr/bin/env python3
"""fonction"""


import numpy as np
import matplotlib.pyplot as plt


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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter function"""
        return self.__W

    @property
    def b(self):
        """getter function"""
        return self.__b

    @property
    def A(self):
        """getter function"""
        return self.__A

    def forward_prop(self, X):
        """
        Calcule la propagation avant du neurone

        Args:
            X: numpy.ndarray avec shape (nx, m) contenant les données d'entrée

        Returns:
            Activation (A): sortie activée du neurone
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """coût"""
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
        self.__W -= alpha * dW
        self.__b -= alpha * db

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
            A = self.forward_prop(X)
            # Descente de gradient pour mettre à jour les poids et le biais
            self.gradient_descent(X, Y, A, alpha)

            # Enregistrer le coût tous les 'step' itérations
            if i % step == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
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
