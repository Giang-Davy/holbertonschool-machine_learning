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

    for i in range(m):
        # Appliquer la convolution sur chaque image
        for idx in range(new_h * new_w):
            y = idx // new_w
            x = idx % new_w
            conv_result[i, y, x] = np.sum(
                images[i, y:y+kh, x:x+kw] * kernel)

    return conv_result
