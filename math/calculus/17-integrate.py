#!/usr/bin/env python3
"""blabla"""

def poly_integral(poly, C=0):
    """
    Args: poly (list of int or float), C (int or float)
    Returns: list or None
    """
    if not isinstance(poly, list) or not all(isinstance(j, (int, float)) for j in poly) or not isinstance(C, (int, float)):
        return None
    
    # Si le polynôme est vide, on retourne une liste vide
    if not poly:
        return []
    
    integral_poly = []
    
    # Ajouter C seulement s'il est différent de 0
    if C != 0 or len(poly) > 1:  # Ajouter C seulement si poly a plus d'un terme
        integral_poly.append(C)
    
    # Calcul de l'intégrale
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)
        integral_poly.append(new_coeff)
    
    return integral_poly
