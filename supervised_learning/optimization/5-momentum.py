#!/usr/bin/env python3
"""fonction"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return v, var
