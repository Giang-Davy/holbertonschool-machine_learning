#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_model(network, filename):
    """sauvegarde modele"""
    network.save(filname)
    return None


def load_model(filename):
    """charge un modele"""
    return K.models.load_model(filename)
