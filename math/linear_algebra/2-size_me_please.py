#!/usr/bin/env python3
"""blabla"""


def matrix_shape(matrix):
    """blabla"""

    shape = []
    while isinstance(matrix, list):  # Vérifier si c'est une list
        shape.append(len(matrix))  # Ajouter la taille de cette dimension
        matrix = matrix[0]  # Descendre au niveau inférieur de la matrice
    return shape
