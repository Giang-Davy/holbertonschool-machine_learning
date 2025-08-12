#!/usr/bin/env python3
"""4-brightness.py"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """change le constrast"""
    brightness = tf.image.random_brightness(image, max_delta)
    return brightness
