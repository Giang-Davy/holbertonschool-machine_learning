#!/usr/bin/env python3
"""2-gru_cell.py"""


import numpy as np


class GRUCell:
    """Gated Recurrent Unit"""
    def __init__(self, i, h, o):
        """initialisation"""
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """forward propagation"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        zt = self.sigmoid(np.dot(conc, self.Wz) + self.bz)
        rt = self.sigmoid(np.dot(conc, self.Wr) + self.br)
        # Calcul de h_next à partir de la porte de réinitialisation
        conc_reset = np.concatenate((rt * h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(conc_reset, self.Wh) + self.bh)
        # Calcul de la sortie finale en combinant h_prev et h_next via zt
        h_next = (1 - zt) * h_prev + zt * h_t
        # Calcul de y
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Applies the softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
