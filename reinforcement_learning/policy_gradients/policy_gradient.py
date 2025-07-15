#!/usr/bin/env python3
"""policy_gradient.py"""


import numpy as np


def policy(matrix, weight):
    """politique simple"""
    z = np.dot(matrix, weight)
    z = np.exp(z)
    S = np.sum(z, axis=1, keepdims=True)
    pi = z / S
    return pi
