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
        """
        Calculate the probability density function (PDF) value for a given
        data point.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        x_centered = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)
        exp_factor = np.exp(
                -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered))

        return float(norm_factor * exp_factor)
    



    
