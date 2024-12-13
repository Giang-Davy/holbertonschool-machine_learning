#!/usr/bin/env python3
"""blabla"""

def poly_integral(poly, C=0):
    """
    Args: poly (list of int or float), C (int or float)
    Returns: list or None
    """
    if not isinstance(poly, list) or not all(isinstance(j, (int, float)) for j in poly) or not isinstance(C, (int, float)):
        return None
    if not poly:  # Vérification si poly est vide
        return None
    integral_poly = [C]
    if len(poly) == 1:
        if poly == [0]:
            return integral_poly
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)  # Convertir en entier si possible
        if new_coeff != 0:  # Omettre les coefficients nuls
            integral_poly.append(new_coeff)
    return integral_poly
