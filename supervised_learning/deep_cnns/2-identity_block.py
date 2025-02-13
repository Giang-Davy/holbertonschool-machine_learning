#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """block_identity"""
    F11, F3, F12 = filters
    initializer = K.initializers.VarianceScaling(scale=2.0)

    convF11 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer=initializer)(A_prev)
    batch1 = K.layers.BatchNormalization(axis=-1)(convF11)
    activation1 = K.layers.ReLU()(batch1)

    convF13 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=initializer)(activation1)
    batch2 = K.layers.BatchNormalization(axis=-1)(convF13)
    activation2 = K.layers.ReLU()(batch2)

    convF12 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer=initializer)(activation2)
    batch3 = K.layers.BatchNormalization(axis=-1)(convF12)
    activation3 = K.layers.ReLU()(batch3)

    output = K.layers.add([activation3, A_prev])

    return output
