#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_model(network, filename):
    """sauvegarde modele"""
    keras.models.save_model(
        network,
        filepath=filename,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )


def load_model(filename):
    """charge un modele"""
    return keras.models.load_model(
        filepath=filename,
        custom_objects=None,
        compile=True
    )
