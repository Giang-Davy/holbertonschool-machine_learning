#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """prediction"""
    prediction = network.predict(data, verbose=False)
    return prediction
