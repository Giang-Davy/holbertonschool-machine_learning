#!/usr/bin/env python3
"""1-crop.py"""


import tensorflow as tf


def crop_image(image, size):
    """rogner une image aléatoirement"""
    crop = tf.image.random_crop(image, size)
    return crop
