#!/usr/bin/env python3
"""fonction"""


def determinant(matrix):
    """Calcul du déterminant d'une matrice carrée"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
    return det


def minor(matrix):
    """Calcul des mineurs d'une matrice carrée"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != matrix_size:
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minors = [[0] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        for j in range(matrix_size):
            sub_matrix = [
                row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minors[i][j] = determinant(sub_matrix)
    return minors
