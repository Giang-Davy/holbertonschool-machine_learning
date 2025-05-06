#!/usr/bin/env python3
"""1-rnn.py"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """forward propagation"""
    t, m, i = X.shape  # Dimensions de X : t (time steps), m (batch size), i (input size)
    h_prev = h_0  # Initialisation de h_prev avec h_0 (état caché initial)
    H = np.zeros((t, m, h_prev.shape[1]))  # Tableau pour stocker tous les états cachés
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))  # Tableau pour stocker toutes les sorties

    for step in range(t):
        x_t = X[step]  # Données d'entrée au temps step
        h_next, y = rnn_cell.forward(h_prev, x_t)  # Propagation avant
        H[step] = h_next  # Stocker le nouvel état caché
        Y[step] = y  # Stocker la sortie
        h_prev = h_next  # Mettre à jour h_prev pour la prochaine itération

    return H, Y

