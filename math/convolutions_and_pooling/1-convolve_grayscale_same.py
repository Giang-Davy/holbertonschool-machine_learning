#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
