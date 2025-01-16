#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """1-chaud"""
    if classes is None:
        classes = len(set(labels))
    one_hot_matrix = K.backend.one_hot(labels, classes)
    # Extraire les valeurs du tenseur
    return K.backend.get_value(one_hot_matrix)
