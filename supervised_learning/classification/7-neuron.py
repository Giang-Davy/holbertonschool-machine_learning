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
        
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Entraîne le neurone"""
        # Validation des paramètres
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
        
        costs = []  # Liste pour stocker les coûts à chaque étape
        for i in range(iterations):
            # Propagation avant pour calculer A
            A = self.forward_prop(X)
            # Calcul du coût
            cost = self.cost(Y, A)
            # Descente de gradient pour mettre à jour les poids et le biais
            self.gradient_descent(X, Y, A, alpha)
            
            # Affichage si verbose est True et à chaque step
            if verbose and (i == 0 or i == iterations - 1 or i % step == 0):
                print(f"Cost after {i} iterations: {cost}")
            
            # Sauvegarde du coût pour le graphique
            if graph and (i == 0 or i == iterations - 1 or i % step == 0):
                costs.append(cost)
        
        # Affichage du graphique si graph est True
        if graph:
            plt.plot(range(0, iterations, step), costs, color="blue")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Retourne l'évaluation après les itérations d'entraînement
        return self.evaluate(X, Y)