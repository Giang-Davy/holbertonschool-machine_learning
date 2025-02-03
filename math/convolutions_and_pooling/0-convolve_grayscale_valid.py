#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """filtre-gris"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    conv_result = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        # Appliquer la convolution sur chaque image
        for j in range(new_w):
            conv_result[:, i, j] = np.sum(
                    images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
                    )

    return conv_result
