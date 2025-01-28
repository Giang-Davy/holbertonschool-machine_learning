#!/usr/bin/env python3
"""fonction"""


import numpy as np


def specificity(confusion):
    "sensiovibilté"
    specificity = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i]) - TP
        TN = np.sum(confusion) - (TP + FN + FP)
        specificity[i] = TN / (TN + FP)

    return specificity
