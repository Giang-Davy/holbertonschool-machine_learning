#!/usr/bin/env python3
"""fonction de pooling"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """effectue le pooling"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1
    pooled_images = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(
                    region, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(
                    region, axis=(1, 2))

    return pooled_images
