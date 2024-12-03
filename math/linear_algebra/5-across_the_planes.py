#!/usr/bin/env python3
"""fonction"""


def add_matrices2D(mat1, mat2):
    """
    Args: ff
    Returns: ff
    Exemple: ff
    """
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    # Ajouter les matrices élément par élément
    add = []
    for i in range(len(mat1)):
        row = []  # Initialiser une nouvelle ligne
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])  # Ajouter les éléments
        add.append(row)  # Ajouter la ligne à la matrice de résultats

    return add
