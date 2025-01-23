#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    dense = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")
    epsilon = 1e-7

    mean, variance = tf.nn.moments(dense, axes=[0])
    normalized = tf.nn.batch_normalization(
        dense, mean, variance, beta, gamma, epsilon)

    if activation:
        return activation(normalized)
    return normalized
