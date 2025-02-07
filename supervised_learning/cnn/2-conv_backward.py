#!/usr/bin/env python3
"""backward pass of a convolutional layer"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Calculates the backward pass of a convolutional layer"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(((h_prev - 1) * sh + kh - h_new) // 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_new) // 2)
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (
            pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    elif padding == "valid":
        pad_h = pad_w = 0
        A_prev_pad = A_prev

    dA_prev = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_prev_slice = A_prev_pad[
                        i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev[
                        i, vert_start:vert_end, horiz_start:horiz_end, :] += W[
                            :, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_prev_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
