#!/usr/bin/env python3
def matrix_shape(matrix):
    # Vérifier si la matrice est une liste de listes (2D)
    if isinstance(matrix[0], list):
        ligne = len(matrix)
        colonne = len(matrix[0])
        return [ligne, colonne]
    else:
        return [len(matrix), 1]  # Si c'est une matrice 1D (liste d'entiers)
