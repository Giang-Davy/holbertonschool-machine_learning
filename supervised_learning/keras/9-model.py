#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    Sauvegarde un modèle Keras complet

    Arguments:
        network: le modèle à sauvegarder
        filename: chemin du fichier où sauvegarder le modèle

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """charge un modele"""
    return K.models.load_model(filename)
