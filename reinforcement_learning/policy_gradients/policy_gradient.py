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


def policy_gradient(state, weight):
    """politique gradiente"""
    politique = policy(state.reshape(1, -1), weight)[0]  # 1D vector
    action = np.random.choice(range(len(politique)), p=politique)
    vecteur = np.zeros_like(politique)
    vecteur[action] = 1
    diff = vecteur - politique
    gradient = np.outer(state, diff)
    return action, gradient
