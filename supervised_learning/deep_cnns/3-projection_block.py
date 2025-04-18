#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """construit un bloc d'identité"""
    F11, F3, F12 = filters

    initializer = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        F11, (1, 1), padding='same',
        strides=s,
        kernel_initializer=initializer)(A_prev)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    conv2 = K.layers.Conv2D(
        F3, (3, 3), strides=1, padding='same',
        kernel_initializer=initializer)(activation1)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation('relu')(batch_norm2)

    conv3 = K.layers.Conv2D(
        F12, (1, 1), strides=1, padding='same',
        kernel_initializer=initializer)(activation2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=s, padding='same',
        kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    output = K.layers.add([batch_norm3, shortcut])
    activated_output = K.layers.Activation('relu')(output)

    return activated_output
