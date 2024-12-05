#!/usr/bin/env python3
"""fonction"""


def np_elementwise(mat1, mat2):
    """
    Returns: ff
    """
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    mul = np.matmul(mat1, mat2)
    div = np.divide(mat1, mat2)
    return add, sub, mul, div
