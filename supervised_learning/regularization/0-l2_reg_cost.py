#!/usr/bin/env python3
"""fonction"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """l2_reg_cost"""
    l2_reg = 0
    for i in range(1, L + 1):
        l2_reg += np.sum(np.square(weights['W' + str(i)]))
    l2_reg_cost = cost + (lambtha / (2 * m)) * l2_reg
    return l2_reg_cost
