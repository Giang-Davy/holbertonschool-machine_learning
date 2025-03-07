#!/usr/bin/env python3
"""Fonction pour le transfert de style neural"""


import numpy as np
import tensorflow as tf


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
        self.gram_style_features = None
        self.content_feature = None
        self.load_model()
        self.generate_features()

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
        if not (
            isinstance(input_layer, (tf.Tensor, tf.Variable))
            and len(input_layer.shape) == 4
        ):
            raise TypeError("input_layer must be a tensor of rank 4")

        # Décomposer les dimensions
        batch_size, h, w, c = input_layer.shape

        # Reshaper l'entrée pour obtenir une matrice (h * w, c)
        reshaped_input = tf.reshape(input_layer, (-1, c))

        # Normaliser l'entrée
        reshaped_input = reshaped_input / tf.sqrt(tf.cast(h * w, tf.float32))

        # Calcul de la matrice de Gram
        gram_matrix = tf.linalg.matmul(
            reshaped_input, reshaped_input, transpose_a=True
        )

        # Ajouter la dimension du batch pour obtenir (1, c, c)
        gram_matrix = tf.expand_dims(gram_matrix, axis=0)

        return gram_matrix

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        # Preprocess the images
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Get the style and content features
        style_outputs = self.model(style_image)
        content_output = self.model(content_image)

        # Calculate the gram matrices for the style features
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]]
        self.content_feature = content_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """couche de style"""
        if not (
            isinstance(style_output, (tf.Tensor, tf.Variable))
            and len(style_output.shape) == 4
        ):
            raise TypeError("style_output must be a tensor of rank 4")

        batch_size, h, w, c = style_output.shape  # Extraire `c`

        if not (
            isinstance(gram_target, (tf.Tensor, tf.Variable))
            and gram_target.shape == (1, c, c)
        ):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        _, c_gram, c_gram2 = gram_target.shape  # Extraire `c_gram`

        if c_gram != c or c_gram2 != c:
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """coût du style total"""
        if not isinstance(style_outputs, list) or len(style_outputs) != len(
            self.style_layers
        ):
            raise TypeError(
                    "style_outputs must be a list with a length of {}"
                    .format(len(self.style_layers))
            )

        total_cost = 0
        weight = 1 / len(self.style_layers)  # Poids égal pour chaque couche

        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            total_cost += weight * self.layer_style_cost(
                style_output, gram_target)

        return total_cost

    def content_cost(self, content_output):
        """coût du contenu"""
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "content_output must be a tensor of shape {}"
                .format(self.content_feature.shape))

        if content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}"
                .format(self.content_feature.shape))

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    def total_cost(self, generated_image):
        """cout total"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(self.content_image.shape))
        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(self.content_image.shape))
        preprocess_generated_image = \
            (tf.keras.applications.
             vgg19.preprocess_input(generated_image * 255))

        # Extraire les caractéristiques de l'image générée
        generated_outputs = self.model(preprocess_generated_image)
        generated_content_output = generated_outputs[-1]
        generated_style_outputs = generated_outputs[:-1]

        J_content = self.content_cost(generated_content_output)
        J_style = self.style_cost(generated_style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return (J, J_content, J_style)
        
    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(self.content_image.shape))
        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(self.content_image.shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
