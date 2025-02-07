#!/usr/bin/env python3
"""fonction"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Backpropagation d'une couche de pooling"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_prev_slice = A_prev[
                        i, vert_start:vert_end,
                        horiz_start:horiz_end, c]

                    if mode == 'max':
                        mask = (
                            a_prev_slice == np.max(a_prev_slice))
                        dA_prev[
                            i, vert_start:vert_end,
                            horiz_start:horiz_end,
                            c] += mask * dA[
                            i, h, w, c]
                    elif mode == 'avg':
                        avg_val = np.mean(a_prev_slice)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += (
                            1. / (kh * kw)) * np.ones_like(a_prev_slice) * dA[
                                i, h, w, c]

    return dA_prev
