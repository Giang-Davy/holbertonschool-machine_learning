#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """1-chaud"""
    if classes is None:
        classes = len(set(labels))
    one_hot_matrix = K.backend.zeros((len(labels), classes))
    for i in range(len(labels)):
        one_hot_matrix[i, labels[i]] = 1
    return one_hot_matrix
