#!/usr/bin/env python3
def matrix_shape(matrix):
    if isinstance(matrix[0], list):
        if isinstance(matrix[0][0], list):
            couche = len(matrix)
            ligne = len(matrix[0])
            colonne = len(matrix[0][0])
            return [couche, ligne, colonne]
        else:
            ligne = len(matrix)
            colonne = len(matrix[0])
            return [ligne, colonne]
    else:
        return [len(matrix), 1]
