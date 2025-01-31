#!/usr/bin/env python3
"""fonction"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """dropout_create_layer"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    couche = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=init)
    dropout = couche(prev)
    dropout = tf.keras.layers.Dropout(rate=1-keep_prob)(dropout, training=training)
    return dropout
