#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """fonction"""
    correct_predictions = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
