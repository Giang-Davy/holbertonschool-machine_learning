#!/usr/bin/env python3
"""fonction"""


def add_matrices2D(mat1, mat2):
    """
    Args: ff
    Returns: ff
    Exemple: ff
    """
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        add = []
    for i in range(len(mat1)):
        row = []
        for k in range(len(mat2)):
            row.append(mat1[i][k] + mat2[i][k])
        add.append(row)
    return add
