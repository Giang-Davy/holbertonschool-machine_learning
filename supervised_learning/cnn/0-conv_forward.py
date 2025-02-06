#!/usr/bin/env python3
"""fonction"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Applique la convolution avec padding et stride"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_h = (h_prev + 2 * ph - kh) // sh + 1
    new_w = (w_prev + 2 * pw - kw) // sw + 1

    # Applique le padding si nécessaire
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Initialisation de la sortie
    Z = np.zeros((m, new_h, new_w, c_new))

    # Calcul de la convolution
    for i in range(new_h):
        for j in range(new_w):
            for c in range(c_new):
                Z[:, i, j, c] = np.sum(
                    A_prev_pad[
                        :, i * sh:i * sh + kh, j * sw:j * sw + kw, :] * W[
                            :, :, :, c],
                    axis=(1, 2, 3)
                )

    # Ajoute le biais à chaque pixel de la sortie
    Z += b

    # Applique la fonction d'activation
    Z = activation(Z)

    return Z
