#!/usr/bin/env python3
"""2-gru_cell.py"""

import numpy as np


class GRUCell:
    """Gated Recurrent Unit"""
    def __init__(self, i, h, o):
        """Initialisation"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.br = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """Forward propagation"""
        # Concatenate h_prev and x_t
        conc = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        zt = self.sigmoid(np.dot(conc, self.Wz) + self.bz)

        # Reset gate
        rt = self.sigmoid(np.dot(conc, self.Wr) + self.br)

        # Candidate hidden state
        conc_reset = np.concatenate((rt * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(conc_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - zt) * h_prev + zt * h_tilde

        # Output
        y = np.dot(h_next, self.Wy) + self.by
        softmax_y = self.softmax(y)

        return h_next, softmax_y

    @staticmethod
    def sigmoid(x):
        """Applique la fonction d'activation sigmoid"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Applique la fonction d'activation softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
