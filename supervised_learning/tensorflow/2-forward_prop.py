#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """fffff"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)

def forward_prop(x, layer_sizes=[], activations=[]):
    """ffff"""
    input_layer = x
    for i in range(len(layer_sizes)):
        input_layer = create_layer(input_layer, layer_sizes[i], activations[i])
    return input_layer
