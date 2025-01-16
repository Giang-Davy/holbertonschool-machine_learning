#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """1-chaud"""
    if classes is None:
        classes = len(set(labels))
    return K.backend.one_hot(labels, classes)
