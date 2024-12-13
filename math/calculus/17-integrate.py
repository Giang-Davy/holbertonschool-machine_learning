#!/usr/bin/env python3
"""Cette fonction calcule l'intégralé par une liste de coefficients."""


def poly_integral(poly, C=0):
    """
    Calcule l'intégrale d'un polynôme représenté par une liste de coefficients.

    Args:
        poly (list): Liste des coeffi où l'index de la liste
                      correspond à la puissance de x pour chaque terme.
        C (int, float): Constante d'intégration, par défaut 0.

    Returns:
        list: Liste des coefficients du polynôme après intégration.
              Si l'entrée est invalide, retourne None.
    """
    if not isinstance(poly, list) or not all(
            isinstance(c, (int, float)) for c in poly):
        return None

    integral_poly = [C]

    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):
            integral_poly.append(int(new_coeff))
        else:
            integral_poly.append(new_coeff)

    return integral_poly
