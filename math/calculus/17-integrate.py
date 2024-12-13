#!/usr/bin/env python3
"""blabla"""

def poly_integral(poly, C=0):
    """
    Args: poly (list of int or float), C (int or float)
    Returns: list or None
    """
    if not isinstance(poly, list) or not all(isinstance(j, (int, float)) for j in poly) or not isinstance(C, (int, float)):
        return None
    integral_poly = []
    
    if C != 0:  # Si C n'est pas zéro, on l'ajoute
        integral_poly.append(C)
    
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):  # Si le coefficient est un entier
            new_coeff = int(new_coeff)
        if new_coeff != 0 or integral_poly:  # Si ce n'est pas zéro, ou si c'est le premier terme
            integral_poly.append(new_coeff)
    
    return integral_poly
