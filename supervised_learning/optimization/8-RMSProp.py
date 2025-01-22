#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """version ameliore de la 7"""
    omptimizer = tf.keras.optimizers.RMSprop(
            learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return omptimizer
