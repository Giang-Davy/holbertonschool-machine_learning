#!/usr/bin/env python3
"""fonction"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    """réseau inception"""
    input = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.VarianceScaling(scale=2.0)

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        activation='relu',
        padding="same",
        strides=2,
        kernel_initializer=initializer)(input)  # Appliquer directement l'entrée à la couche Conv2D
    
    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same")(conv1)  # Appliquer la couche MaxPooling après conv1

    output_inception = inception_block(pool1, [64, 96, 128, 16, 32, 32])  # Passer pool1 à inception_block

    model = K.models.Model(inputs=input, outputs=output_inception)  # Créer le modèle avec l'entrée et la sortie spécifiées

    return model
