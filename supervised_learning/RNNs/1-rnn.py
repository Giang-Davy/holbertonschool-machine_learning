#!/usr/bin/env python3
"""1-rnn.py"""

import numpy as np

def rnn(rnn_cell, X, h_0):
    """Forward propagation"""
    t, m, i = X.shape
    h_prev = h_0
    H = [h_0]  # Inclure l'état initial h_0 dans H
    Y = []
    for k in range(t):
        x_t = X[k]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H.append(h_next)  # Ajouter h_next à H
        Y.append(y)       # Ajouter y à Y
        h_prev = h_next   # Mettre à jour h_prev pour l'itération suivante
    return np.array(H), np.array(Y)
