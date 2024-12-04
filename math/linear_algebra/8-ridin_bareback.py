#!/usr/bin/env python3
"""fonction"""


def mat_mul(mat1, mat2):
    """
    Args: ff
    Returns: ff
    Exemple: ff
    """
    add = []

    # Parcours des lignes de mat1
    for i in range(len(mat1)):
        new_row = []  # Crée une nouvelle ligne vide
        # Parcours des colonnes de mat2
        for k in range(len(mat2[0])):
            somme = 0
            for j in range(len(mat1[0])):
                somme += mat1[i][j] * mat2[j][k]
            new_row.append(somme)
        add.append(new_row)
    return add
