#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """filtre-gris"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    conv_result = np.zeros((m, h, w))

    # Appliquer le padding sur les images
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for i in range(h):
        for j in range(w):
            conv_result[:, i, j] = np.sum(
                    padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
                    )

    return conv_result
