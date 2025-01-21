#!/usr/bin/env python3
"""fonction"""


import numpy as np


def moving_average(data, beta):
    """
    Calcule la moyenne mobile exponentielle avec correction du biais
    """
    v_prev = 0
    moving_averages = []

    for t, x_t in enumerate(data, 1):
        v_t = beta * v_prev + (1 - beta) * x_t
        v_t_corrected = v_t / (1 - beta**t)
        moving_averages.append(v_t_corrected)
        v_prev = v_t

    return moving_averages
