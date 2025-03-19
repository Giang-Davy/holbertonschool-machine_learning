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

    if (row, col) == (1, 1):
        return matrix[0][0]

    if (row, col) == (2, 2):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant_value = 0
    sign = 1
    matrix_size = len(matrix)
    for col in range(matrix_size):
        pivot_element = matrix[0][col]
        minor_matrix = []
        for row in range(1, matrix_size):
            new_row = [matrix[row][c] for c in range(matrix_size) if c != col]
            minor_matrix.append(new_row)
        determinant_value += pivot_element * sign * determinant(minor_matrix)
        sign *= -1
    return determinant_value
