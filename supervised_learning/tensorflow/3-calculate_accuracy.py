#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calcule"""
    correct_predictions = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
