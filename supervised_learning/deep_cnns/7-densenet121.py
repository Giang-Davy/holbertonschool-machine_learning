#!/usr/bin/env python3
"""Contains the function densenet121."""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture."""
    input_1 = K.layers.Input(shape=(224, 224, 3))
    batch_normalization = K.layers.BatchNormalization(axis=-1)(input_1)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv7x7 = K.layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(activation)

    max_pooling2d = K.layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding='same'
    )(conv7x7)

    dense_block_1, nb_filters = dense_block(
        max_pooling2d, 64,
        growth_rate, 6
    )
    transition_layer_1, nb_filters = transition_layer(
        dense_block_1,
        nb_filters,
        compression
    )

    dense_block_2, nb_filters = dense_block(
        transition_layer_1,
        nb_filters,
        growth_rate, 12
    )
    transition_layer_2,  nb_filters = transition_layer(
        dense_block_2,
        nb_filters,
        compression
    )

    dense_block_3, nb_filters = dense_block(
        transition_layer_2,
        nb_filters,
        growth_rate,
        24
    )
    transition_layer_3, nb_filters = transition_layer(
        dense_block_3,
        nb_filters,
        compression
    )

    dense_block_4, nb_filters = dense_block(
        transition_layer_3,
        nb_filters,
        growth_rate,
        16
    )

    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(dense_block_4)

    dense = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(average_pooling2d)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model









