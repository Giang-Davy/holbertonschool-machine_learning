#!/usr/bin/env python3
"""Fonction pour le transfert de style neural"""


import tensorflow as tf
import numpy as np


class NST:
    """Neurone-Style-Transfert"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialisation"""
        if not (
            isinstance(style_image, np.ndarray) and style_image.shape[-1] == 3
        ):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        else:
            self.style_image = self.scale_image(style_image)

        if not (
            isinstance(content_image, np.ndarray)
            and content_image.shape[-1] == 3
        ):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        else:
            self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        height, width, _ = image.shape

        if height > width:
            h_new = 512
            w_new = int((width / height) * 512)
        else:
            w_new = 512
            h_new = int((height / width) * 512)

        resized_image = tf.image.resize(
            image, (h_new, w_new), method='bicubic')

        # Convertir en float32 et normaliser
        resized_image = tf.cast(resized_image, tf.float32) / 255.0
        resized_image = tf.clip_by_value(resized_image, 0.0, 1.0)

        return resized_image[tf.newaxis, ...]

    def load_model(self):
        """Creates the model used to calculate cost"""
        # Load the VGG19 model
        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        modelVGG19.trainable = False

        # Select the layers
        selected_layers = self.style_layers + [self.content_layer]
        outputs = [
            modelVGG19.get_layer(name).output for name in selected_layers]

        # Construct the model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # Replace MaxPooling layers with AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model(
            'vgg_base.h5', custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """calcul matrix"""
        if not (isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # Décomposer les dimensions
        batch_size, h, w, c = input_layer.shape

        # Reshaper l'entrée pour obtenir une matrice (h * w, c)
        reshaped_input = tf.reshape(input_layer, (-1, c))

        # Normaliser l'entrée
        reshaped_input = reshaped_input / tf.sqrt(tf.cast(h * w, tf.float32))

        # Calcul de la matrice de Gram
        gram_matrix = tf.linalg.matmul(
            reshaped_input, reshaped_input, transpose_a=True)

        # Ajouter la dimension du batch pour obtenir (1, c, c)
        gram_matrix = tf.expand_dims(gram_matrix, axis=0)

        return gram_matrix
