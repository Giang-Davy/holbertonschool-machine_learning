#!/usr/bin/env python3
"""fonction"""


import numpy as np


def precision(confusion):
    """precision"""
    precision = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        FP = np.sum(confusion[:, i]) - TP
        precision[i] = TP / (TP + FP)

    return precision
