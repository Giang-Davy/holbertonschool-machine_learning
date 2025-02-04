#!/usr/bin/env python3
"""fonction"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    
    oh = (padded_h - kh) // sh + 1
    ow = (padded_w - kw) // sw + 1
    
    output = np.zeros((m, oh, ow))
    
    for i in range(0, oh):
        for j in range(0, ow):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )
    
    return output
