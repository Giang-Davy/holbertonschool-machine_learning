#!/usr/bin/env python3
"""fonction"""


import numpy as np


def sensitivity(confusion):
    "sensiovibilté"
    sensitivity = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        FN = np.sum(confusion[i]) - TP
        sensitivity[i] = TP / (TP + FN)

    return sensitivity
