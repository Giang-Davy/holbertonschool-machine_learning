#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def save_config(network, filename):
    """Sauvegarde la configuration du modèle en JSON"""
    model_json = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(model_json)
    return None


def load_config(filename):
    """Charge la configuration du modèle à partir d'un fichier JSON"""
    with open(filename, 'r') as json_file:
        model_json = json_file.read()
    model = K.models.model_from_json(model_json)
    return model
