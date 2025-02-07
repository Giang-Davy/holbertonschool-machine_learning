import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    if padding == "same":
        pad_h = ((h_new - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_new - 1) * sw + kw - w_prev) // 2
    else:
        pad_h = pad_w = 0

    A_prev_padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
    dA_prev_padded = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = dA_prev_padded[i, pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_padded[i, :, :, :]

    return dA_prev, dW, db
