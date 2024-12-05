#!/usr/bin/env python3
"""fonction"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Args: ff
    Returns: ff
    """
    if axis == 0:
        result_0 = np.concatenate((mat1, mat2), axis=0)
        return result_0
    if axis == 1:
        result_1 = np.concatenate((mat1, mat2), axis=0)
        return result_1
