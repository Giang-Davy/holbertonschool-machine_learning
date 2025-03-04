#!/usr/bin/env python3
"""Fonction pour le transfert de style neural"""

import tensorflow as tf
import numpy as np




class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        height, width, _ = image.shape

        if height > width:
            h_new = 512
            w_new = int((width / height) * 512)
        else:
            w_new = 512
            h_new = int((height / width) * 512)

        resized_image = tf.image.resize(image, (h_new, w_new), method='bicubic')

        # Convertir en float32 et normaliser
        resized_image = tf.cast(resized_image, tf.float32) / 255.0  

        return resized_image[tf.newaxis, ...]
