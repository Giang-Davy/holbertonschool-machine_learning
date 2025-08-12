#!/usr/bin/env python3
"""5-hue.py"""


import tensorflow as tf


def change_hue(image, delta):
    """change le constrast"""
    hue = tf.image.adjust_hue(image, delta)
    return hue
