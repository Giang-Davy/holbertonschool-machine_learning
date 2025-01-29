#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    prev: tensor
        The output of the previous layer.
    n: int
        The number of nodes the new layer should have.
    activation: function
        The activation function to use for the layer.
    lambtha: float
        The L2 regularization parameter.

    Returns:
    tensor: The output of the new layer with L2 regularization.
    """
    return tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )(prev)
