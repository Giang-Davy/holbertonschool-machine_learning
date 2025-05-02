#!/usr/bin/env python3
"""3-generate_faces.py"""

import tensorflow as tf
from tensorflow import keras

def convolutional_GenDiscr():
    def generator():
        model = keras.models.Sequential([
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(2048, activation='tanh'),  # Augmenté à 2048 neurones
            keras.layers.Reshape((2, 2, 512)),  # Reshape en (2, 2, 512)
            keras.layers.UpSampling2D(size=(2, 2)),  # Passe de 2x2 à 4x4
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh"),
            keras.layers.UpSampling2D(size=(2, 2)),  # Passe de 4x4 à 8x8
            keras.layers.Conv2D(16, (3, 3), padding="same"),  # Ajout d'une étape intermédiaire avec 16 filtres
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh"),
            keras.layers.UpSampling2D(size=(2, 2)),  # Passe de 8x8 à 16x16
            keras.layers.Conv2D(1, (3, 3), padding="same"),  # Pas d'UpSampling supplémentaire
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh")
        ], name="generator")
        return model
    
    def discriminator():
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 1)),  # Changé à 16x16
            keras.layers.Conv2D(32, (3, 3), padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("tanh"),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("tanh"),
            keras.layers.Conv2D(128, (3, 3), padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("tanh"),
            keras.layers.Conv2D(256, (3, 3), padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("tanh"),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ], name="discriminator")
        return model
    return generator(), discriminator()
