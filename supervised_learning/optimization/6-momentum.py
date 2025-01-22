#!/usr/bin/env python3
"""fonction"""

import tensorflow as tf
import numpy as np


def create_momentum_op(alpha, beta1):
    """momentum ameliore"""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
