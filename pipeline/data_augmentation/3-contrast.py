#!/usr/bin/env python3
"""2-rotate.py"""


import tensorflow as tf


def change_contrast(image, lower, upper):
    """change le constrast"""
    contrast = tf.image.random_contrast(image, lower, upper)
    return contrast
