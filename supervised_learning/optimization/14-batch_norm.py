#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    
    # Dense layer without activation
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)(prev)
    
    # Batch normalization
    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)(dense)
    
    # Apply the activation function
    if activation:
        output = activation(batch_norm)
    else:
        output = batch_norm
    
    return output
