#!/usr/bin/env python3
"""fonction"""


def matrix_transpose(matrix):
    """
    Args: ff
    Returns : ff
    Exemple : ff
    """
    transposed = []
    for i in range(len(matrix[0])):  # Itère sur les colonne
        colo = []  # Liste pour la nouvelle ligne
        for row in matrix:  # Parcourt chaque e originale
            colo.append(row[i])  # Ajoute l'é i à la nouvelle ligne
        transposed.append(colo)  # Ajoute la le à la matrice finale
    return transposed
