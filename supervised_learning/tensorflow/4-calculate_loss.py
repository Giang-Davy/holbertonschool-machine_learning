#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calcule perte"""
    loss = tf.compat.v1.losses.softmax_cross_entropy(
          onehot_labels=y, logits=y_pred)
    return loss
