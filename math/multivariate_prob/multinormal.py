#!/usr/bin/env python3
"""multinormal.py"""


import numpy as np


class MultiNormal:
    """classe multinormal"""
    def __init__(self, data):
        """initialisation"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance
        X_centered = data - mean
        cov = np.dot(X_centered, X_centered.T) / (n - 1)

        self.mean = mean
        self.cov = cov

    def pdf(self, x):
        """calculer le pdf"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        X_centered = x - self.mean

        # Correct PDF formula
        exponent = -0.5 * np.dot(np.dot(X_centered.T, inv_cov), X_centered)
        pdf_value = (
            1 / np.sqrt((2 * np.pi) ** d * det_cov)) * np.exp(exponent)

        return float(pdf_value)  # Return as a scalar
