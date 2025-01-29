#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    init_weights = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg")
    
    l2_reg = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_reg,
        kernel_initializer=init_weights)
        
    return layer(prev)
