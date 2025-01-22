#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    optimize = tf.keras.optimizers.Adam(
            learning_rate=alpha, epsilon=epsilon, beta_1=beta1, beta_2=beta2)
    return optimize
