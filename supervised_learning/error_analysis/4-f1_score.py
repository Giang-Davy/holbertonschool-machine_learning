#!/usr/bin/env python3
"""fonction"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """f1_score"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1 = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        f1[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])

    return f1
