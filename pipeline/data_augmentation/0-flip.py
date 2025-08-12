#!/usr/bin/env python3
"""0-flip.py"""


import tensorflow as tf


def flip_image(image):
    """tourner une image horizontalement"""
    flip = tf.image.flip_left_right(image)
    return flip
