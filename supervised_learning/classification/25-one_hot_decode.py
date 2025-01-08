#!/usr/bin/env python3
"""fonction"""


import numpy as np


def one_hot_decode(one_hot):
    """ff"""
    # Vérification de la validité de l'entrée
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    if not np.all(np.sum(one_hot, axis=0) == 1):
        return None

    # Retourne les indices des valeurs 1 dans chaque colonne (les labels)
    return np.argmax(one_hot, axis=0)
