#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    initializer = K.initializers.VarianceScaling(scale=2.0)

    convF1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(A_prev)

    convF3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(A_prev)

    convF3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(convF3R)

    convF5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(A_prev)

    convF5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(convF5R)

    poolFPP = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same")(A_prev)

    poolFPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        activation='relu',
        padding="same",
        kernel_initializer=initializer)(poolFPP)
    concatenated_output = K.layers.Concatenate()([
        convF1, convF3, convF5, poolFPP])

    return concatenated_output
