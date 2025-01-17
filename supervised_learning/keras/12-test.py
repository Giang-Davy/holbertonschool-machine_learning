#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """test de modele"""
    loss, accuracy = model.evaluate(network, data, labels)
    return loss, accuracy
