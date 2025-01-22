#!/usr/bin/env python3
"""fonction"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update variables using the Adam optimization algorithm."""
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment
    v_corrected = v / (1 - beta1 ** t)

    # Compute bias-corrected second moment
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)

    return var, v, s
