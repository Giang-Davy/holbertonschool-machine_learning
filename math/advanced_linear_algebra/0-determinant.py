#!/usr/bin/env python3
"""fonction"""


def determinant(matrix):
    """determinant d'une matrice"""
    # Vérifie si la matrice est une liste de listes
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 and matrix_size == 1:
            return 1
        if len(row) != matrix_size:
            raise ValueError("matrix must be a square matrix")

    # Special case: empty matrix (0x0)
    if len(matrix[0]) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
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
