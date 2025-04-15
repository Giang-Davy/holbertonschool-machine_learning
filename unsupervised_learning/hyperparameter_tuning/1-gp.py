#!/usr/bin/env python3
"""0-gp.py"""


import numpy as np


class GaussianProcess:
    """class de Gauss"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """initialisation"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """kernel"""

        sqdist = np.sum(
            X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """predict"""
        Ks = self.kernel(self.X, X_s)
        Kss = self.kernel(X_s, X_s)
        mu = Ks.T @ np.linalg.inv(self.K) @ self.Y
        sigma = np.diag(Kss - Ks.T @ np.linalg.inv(self.K) @ Ks)

        return mu.reshape(-1), sigma
