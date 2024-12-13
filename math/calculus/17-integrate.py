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
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)  # Convertir en entier si possible
        integral_poly.append(new_coeff)
    
    # Ajouter des zéros dans la sortie pour correspondre à l'attendu (s'il y a des zéros explicites à insérer)
    degree = len(poly)
    for i in range(degree + 1, len(integral_poly)):
        integral_poly.insert(i, 0)

    return integral_poly
