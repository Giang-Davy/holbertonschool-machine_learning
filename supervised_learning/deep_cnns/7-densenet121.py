#!/usr/bin/env python3
"""Deep neural architecture"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    img_input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    L1 = K.layers.BatchNormalization(axis=3)(img_input)
    L1 = K.layers.Activation('relu')(L1)
    L1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         kernel_initializer=init)(L1)
    L1 = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(L1)
    L1, filters = dense_block(L1, 64, growth_rate, 6)
    L1, filters = transition_layer(L1, filters, compression)
    L1, filters = dense_block(L1, filters, growth_rate, 12)
    L1, filters = transition_layer(L1, filters, compression)
    L1, filters = dense_block(L1, filters, growth_rate, 24)
    L1, filters = transition_layer(L1, filters, compression)
    L1, filters = dense_block(L1, filters, growth_rate, 16)
    L1 = K.layers.AvgPool2D((7, 7), padding='same')(L1)
    L1 = K.layers.Dense(1000, activation='softmax')(L1)
    model = K.Model(img_input, L1)
    return model
