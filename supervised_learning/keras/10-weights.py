#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """sauvegarde poid"""
    network.save_weights(filename, save_format='keras')
    return None


def load_weights(network, filename):
    """charge poid"""
    network.load_weights(filename)
