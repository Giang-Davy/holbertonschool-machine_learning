#!/usr/bin/env python3
"""fonction"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """pool en avant"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    new_h = (h_prev - kh) // sh + 1
    new_w = (w_prev - kw) // sw + 1
    conv_result = np.zeros((m, new_h, new_w, c_prev))

    for i in range(new_h):
        for j in range(new_w):
            region = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                conv_result[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                conv_result[:, i, j, :] = np.mean(region, axis=(1, 2))

    return conv_result
