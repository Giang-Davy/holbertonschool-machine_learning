#!/usr/bin/env python3
"""4-deep_rnn.py"""

import numpy as np

def deep_rnn(rnn_cells, X, h_0):
    """Forward propagation for a deep RNN"""
    t, m, i = X.shape  # t: time steps, m: batch size, i: input size
    l, _, h = h_0.shape  # l: number of layers, h: hidden state size
    H = np.zeros((t + 1, l, m, h))  # Hidden states for all time steps and layers
    H[0] = h_0  # Initialize with the initial hidden states

    Y = []  # List to store outputs at each time step

    for time_step in range(t):
        x_t = X[time_step]  # Input at the current time step
        h_prev_layer = h_0[:, :, :]  # Initialize with the hidden states for all layers

        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(h_prev_layer[layer], x_t)
            H[time_step + 1, layer] = h_next  # Store the hidden state
            h_prev_layer[layer] = h_next  # Update the hidden state for the current layer
            x_t = h_next  # Pass the hidden state as input to the next layer

        Y.append(y)  # Append the output of the last layer

    return H, np.array(Y)
