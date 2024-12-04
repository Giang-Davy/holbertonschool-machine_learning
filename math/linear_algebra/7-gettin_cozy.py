#!/usr/bin/env python3
"""fonction"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Args: ff
    Returns: ff
    Exemple: ff
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for i in range(len(mat1)):
            new = mat1[i] + mat2[i]
            new_matrix.append(new)
        return new_matrix
