#!/usr/bin/env python3
"""fonction"""


def mat_mul(mat1, mat2):
    """
    Args: ff
    Returns: ff
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for k in range(len(mat2[0])):
            somme = 0
            for j in range(len(mat1[0])):
                somme += mat1[i][j] * mat2[j][k]
            row.append(somme)
        result.append(row)
    return result
