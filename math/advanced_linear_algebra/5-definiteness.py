#!/usr/bin/env python3
"""fonction"""


import numpy as np


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
    matrix_size = len(matrix)
    if len(matrix) == 1:
        return [[1]]

    minors = [[0] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        for j in range(matrix_size):
            sub_matrix = [
                row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minors[i][j] = determinant(sub_matrix)
    return minors


def cofactor(matrix):
    """cofacteur pour inverser les signes"""
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

    minors = minor(matrix)
    cofactor = [[0] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        for j in range(matrix_size):
            cofactor[i][j] = minors[i][j] * ((-1) ** (i + j))
    return cofactor


def adjugate(matrix):
    """adjointe de matrice qui permet de changer les places des valeurs"""
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
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [list(row) for row in zip(*cofactor_matrix)]
    return adjugate_matrix


def inverse(matrix):
    """Qui permet de calculer leur inverse"""
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
    determinant_number = determinant(matrix)
    if determinant_number == 0:
        return None
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = [[
        adjugate_matrix[i][j]
        / determinant_number for j in range(matrix_size)] for i in range(
            matrix_size)]
    return inverse_matrix


def definiteness(matrix):
    """définir une matrice"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.array_equal(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
