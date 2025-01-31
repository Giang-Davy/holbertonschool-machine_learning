#!/usr/bin/env python3
"""fonction"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """dropout_create_layer"""
    couche = tf.keras.layers.Dense(n, activation=activation)(prev)
    dropout = tf.keras.layers.Dropout(rate=1-keep_prob)
    return dropout(couche, training=training)
