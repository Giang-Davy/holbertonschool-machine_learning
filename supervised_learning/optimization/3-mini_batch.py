#!/usr/bin/env python3
"""fonction"""


import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Create mini-batches for mini-batch gradient descent.
    """
    # Étape 1 : Mélanger les données
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]  # Nombre total de points de données

    # Étape 2 : Initialiser la liste des mini-batches
    mini_batches = []

    # Étape 3 : Créer les mini-batches complets
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    # Étape 4 : Retourner la liste des mini-batches
    return mini_batches
