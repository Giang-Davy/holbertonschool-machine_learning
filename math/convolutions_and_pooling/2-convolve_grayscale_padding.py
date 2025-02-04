#!/usr/bin/env python3
"""fonction"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """padding_custom"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    padded_images = np.pad(
        images, ((0,), (pad_h,), (pad_w,)), mode='constant', constant_values=0)
    new_h = h + 2 * pad_h - kh
    new_w = w + 2 * pad_w - kw
    conv_result = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            conv_result[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return conv_result
