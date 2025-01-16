#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """optimisation"""
    K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2)

    network.compile(
            optimizer=K.optimizers.Adam(learning_rate=alpha,
                                        beta_1=beta1,
                                        beta_2=beta2),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    return None
