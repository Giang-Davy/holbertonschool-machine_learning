#!/usr/bin/env python3
"""
Convolution Back Propagation
"""

import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):

    # Extract dimensions from the input shapes
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Determine padding size
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5)

    # Apply padding to A_prev
    A_prev_pad = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)], mode='constant')

    # Initialize gradients
    dA_prev = np.zeros(A_prev.shape)
    dA_prev_pad = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Calculate the bias gradient
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = dA_prev_pad[i, ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]

    return dA_prev, dW, db
