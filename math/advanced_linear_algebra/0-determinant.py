#!/usr/bin/env python3
"""fonction"""


def determinant(matrix):
    """determinant d'une matrice"""
    # Vérifie si la matrice est une liste de listes
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # Special case: empty matrix (0x0)
    if len(matrix[0]) == 0:
        return 1

    # test if matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    row = len(matrix)
    col = len(matrix[0])

    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    if (row, col) == (3, 3):
        return (matrix[0][0] * (
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (
                    matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (
                    matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
