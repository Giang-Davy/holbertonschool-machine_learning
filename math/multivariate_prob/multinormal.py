#!/usr/bin/env python3
"""multinormal.py"""


import numpy as np


class MultiNormal:
    """classe multinormal"""
    def __init__(self, data):
        """initialisation"""
        if not isinstance(data, np.ndarray):
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
