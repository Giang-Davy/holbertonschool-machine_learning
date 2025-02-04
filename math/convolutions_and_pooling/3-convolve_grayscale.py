#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """convolution-gris"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        pad_h, pad_w = 0, 0
        new_h = (h - kh) // sh + 1
        new_w = (w - kw) // sw + 1
    elif padding == 'same':
        pad_h = kh // 2
        pad_w = kw // 2
        new_h = (h + 2 * pad_h - kh) // sh + 1
        new_w = (w + 2 * pad_w - kw) // sw + 1
    else:
        pad_h, pad_w = padding
        new_h = (h + 2 * pad_h - kh) // sh + 1
        new_w = (w + 2 * pad_w - kw) // sw + 1

    padded_images = np.pad(
        images, ((0,), (pad_h,), (pad_w,)), mode='constant', constant_values=0)

    conv_result = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            conv_result[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return conv_result
