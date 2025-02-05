#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """convolution-gris"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    conv_result = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            conv_result[:, i, j] = np.sum(
                padded_images[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2, 3)
            )

    return conv_result
