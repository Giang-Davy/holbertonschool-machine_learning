#!/usr/bin/env python3
"""2-rotate.py"""


import tensorflow as tf


def rotate_image(image):
    """fais une roration d'une image"""
    rotate = tf.image.rot90(image)
    return rotate
