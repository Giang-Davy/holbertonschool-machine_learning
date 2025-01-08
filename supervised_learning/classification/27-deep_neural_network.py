#!/usr/bin/env python3
"""
module
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Classe DeepNeuralNetwork qui définit un réseau de neurones profond
    réalisant une classification binaire
    """

    def __init__(self, nx, layers):
        """
        initialisation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """
        getter
        """
        return self.__L

    @property
    def cache(self):
        """
        getter
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter
        """
        return self.__weights

    def forward_prop(self, X):
        """
        calcul
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                self.__cache[Akey] = self.sigmoid(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def sigmoid(self, z):
        """
        formule
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """
        formule2
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def cost(self, Y, A):
        """
        cout
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        evaluer
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        dfff
        """
        m = Y.shape[1]
        Al = cache["A{}".format(self.__L)]
        dAl = (-Y / Al) + (1 - Y)/(1 - Al)

        for i in reversed(range(1, self.__L + 1)):
            wkey = "W{}".format(i)
            bkey = "b{}".format(i)
            Al = cache["A{}".format(i)]
            Al1 = cache["A{}".format(i - 1)]
            g = Al * (1 - Al)
            dZ = np.multiply(dAl, g)
            dW = np.matmul(dZ, Al1.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights["W{}".format(i)]
            dAl = np.matmul(W.T, dZ)

            self.__weights[wkey] = self.__weights[wkey] - alpha * dW
            self.__weights[bkey] = self.__weights[bkey] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        entraienement
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
            cost_list.append(cost)
            step_list.append(i)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        sauvegarde
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        charger
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
