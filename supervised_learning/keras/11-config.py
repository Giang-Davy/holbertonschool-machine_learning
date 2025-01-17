#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_config(network, filename):
    """sauvegarde JSON"""
    network.to_json(filename)


def load_config(filename):
    """charge JSON"""
    return keras.models.model_from_json(filename)
