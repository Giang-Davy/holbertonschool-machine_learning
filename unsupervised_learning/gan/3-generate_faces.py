#!/usr/bin/env python3
"""3-generate_faces.py"""


import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """contient une fonctione du generateur et discriminateur"""
    def generator():
        """generateur"""
        model = keras.models.Sequential([
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(2048, activation='tanh'),
            keras.layers.Reshape((2, 2, 512)),
            keras.layers.UpSampling2D(size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh"),
            keras.layers.UpSampling2D(size=(2, 2)),
            keras.layers.Conv2D(16, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh"),
            keras.layers.UpSampling2D(size=(2, 2)),
            keras.layers.Conv2D(1, (3, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("tanh")
        ], name="generator")
        return model

    def discriminator():
        """discriminateur"""
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 1)),
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
