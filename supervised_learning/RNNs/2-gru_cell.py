#!/usr/bin/env python3
"""2-gru_cell.py"""


import numpy as np


class GRUCell:
    """Gated  Recurrent unit"""
    def __init__(self, i, h, o):
        """initialisation"""
        np.random.seed(0)
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """forward propagration"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        zt = 1 / (1 + np.exp(-np.dot(conc, self.Wz) + self.bz))
        rt = 1 / (1 + np.exp(-np.dot(conc, self.Wr) + self.br))
        # Calcul de h_next à partir de la porte de réinitialisation
        conc_reset = np.concatenate((rt * h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(conc_reset, self.Wh) + self.bh)
        # Calcul de la sortie finale en combinant h_prev et h_next via zt
        ht = (1 - zt) * h_prev + zt * h_next
        # Calcul de y
        y = np.dot(ht, self.Wy) + self.by
        # Application de softmax
        max_y = np.max(y, axis=1, keepdims=True)
        exp_y = np.exp(y - max_y)
        sum_y = np.sum(exp_y, axis=1, keepdims=True)
        softmax_y = exp_y / sum_y

        return h_next, softmax_y
