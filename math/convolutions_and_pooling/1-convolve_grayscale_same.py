#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """filtre-gris"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcul du padding pour les noyaux impairs et pairs
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Appliquer le padding sur les images
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialisation du résultat de convolution
    conv_result = np.zeros((m, h, w))

    # Appliquer la convolution sur chaque image
    for i in range(h):
        for j in range(w):
            conv_result[:, i, j] = np.sum(
                    padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
                    )

    return conv_result
