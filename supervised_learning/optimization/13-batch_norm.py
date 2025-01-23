#!/usr/bin/env python3
"""fonction"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Performs batch normalization on unactivated output Z."""
    # Calculate the mean and variance of Z along axis 0
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)

    # Normalize Z using the mean and variance
    Z_normalized = (Z - mean) / ((variance + epsilon) ** 0.5)

    # Scale and shift the normalized Z using gamma and beta
    Z_batch_normalized = gamma * Z_normalized + beta

    return Z_batch_normalized
