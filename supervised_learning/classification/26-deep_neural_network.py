#!/usr/bin/env python3
"""
Module définissant un réseau de neurones profond
pour la classification binaire.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Classe définissant un réseau de neurones profond
    pour la classification binaire.

    Attributs privés :
        __L (int): Nombre de couches dans le réseau de neurones.
        __cache (dict): Dictionnaire pour stocker les valeurs intermédiaires.
        __weights (dict): Dictionnaire pour stocker les poids
        et biais du réseau de neurones.
    """

    def __init__(self, nx, layers):
        """
        Initialise un réseau de neurones profond.

        Args:
            nx (int): Nombre de caractéristiques d'entrée.
            layers (list): Liste contenant le nombre de nœuds
            pour chaque couche du réseau.

        Raises:
            TypeError: Si nx n'est pas un entier.
            ValueError: Si nx est inférieur à 1.
            TypeError: Si layers n'est pas une liste de nombres positifs.
        """
        # Vérifications des paramètres
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialisation des poids et biais
        for i in range(self.__L):
            layer_num = i + 1
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights[f'W{layer_num}'] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights[f'W{layer_num}'] = (np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]))
            self.__weights[f'b{layer_num}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Retourne le nombre de couches du réseau de neurones."""
        return self.__L

    @property
    def cache(self):
        """Retourne le dictionnaire de cache du réseau de neurones."""
        return self.__cache

    @property
    def weights(self):
        """Retourne le dictionnaire des poids et biais
        du réseau de neurones."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau de neurones

        Args:
            X (numpy.ndarray): données d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques
                m est le nombre d'exemples

        Returns:
            tuple: (sortie du réseau, cache)
        """
        self.__cache['A0'] = X

        for index_couche in range(1, self.__L + 1):
            W = self.__weights[f'W{index_couche}']
            b = self.__weights[f'b{index_couche}']
            A_prev = self.__cache[f'A{index_couche-1}']

            Z = np.matmul(W, A_prev) + b
            self.__cache[f'A{index_couche}'] = 1 / (1 + np.exp(-Z))

        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """
        Calcule le coût du modèle avec la régression logistique

        Args:
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)
            A (numpy.ndarray): sortie activée du réseau de forme (1, m)

        Returns:
            float: Le coût
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Évalue les prédictions du réseau de neurones

        Args:
            X (numpy.ndarray): données d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques
                m est le nombre d'exemples
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)

        Returns:
            tuple: (prédictions, coût)
                prédictions est une matrice de forme (1, m)
                coût est un scalaire
        """
        # Propagation avant pour obtenir les prédictions
        A, _ = self.forward_prop(X)

        # Calcul du coût
        cost = self.cost(Y, A)

        # Conversion des probabilités en prédictions binaires
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calcule une passe de descente de gradient sur le réseau de neurones

        Args:
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)
            cache (dict): dictionnaire contenant toutes les valeurs
                         intermédiaires du réseau
            alpha (float): taux d'apprentissage
        Lexique perso:
        'd' représente la dérivée partielle
                'Z' représente la sortie avant l'activation
                Par convention 'dZ' représente la dérivée de
        la fonction de coût par rapport à Z
        """
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entraîne le réseau de neurones profond

        Args:
            X (numpy.ndarray): données d'entrée de forme (nx, m)
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)
            iterations (int): nombre d'itérations d'entraînement
            alpha (float): taux d'apprentissage
            verbose (bool): affiche le coût pendant l'entraînement
            graph (bool): affiche le graphique d'apprentissage
            step (int): pas entre les affichages

        Returns:
            tuple: (prédictions finales, coût final)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            # Propagation avant
            A, cache = self.forward_prop(X)

            # Affichage du coût
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                # Descente de gradient
                self.gradient_descent(Y, cache, alpha)

        # Affichage du graphique
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Sauvegarde l'instance du réseau de neurones dans un fichier

        Args:
            filename (str): nom du fichier de sauvegarde

        Notes:
            Si filename n'a pas l'extension .pkl, elle est ajoutée
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Charge une instance de DeepNeuralNetwork depuis un fichier

        Args:
            filename (str): nom du fichier à charger

        Returns:
            DeepNeuralNetwork: l'instance chargée, None si erreur
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
