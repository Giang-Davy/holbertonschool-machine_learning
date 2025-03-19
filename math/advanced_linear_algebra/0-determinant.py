#!/usr/bin/env python3
"""fonction"""


def sub_matrix(matrix, i):
    """
    Creates a submatrix by removing the first row and the i-th column
    """
    if not matrix:
        return []

    matrix2 = []
    for row in matrix[1:]:  # Skip the first row
        matrix2.append(row[:i] + row[i + 1:])  # Remove the i-th column

    return matrix2

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

    det = 0
    for i in range(len(matrix[0])):
        det += (
                ((-1) ** i) * matrix[0][i] * determinant(
                    sub_matrix(matrix, i)
                    )
                )
    return det
