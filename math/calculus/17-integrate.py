#!/usr/bin/env python3
"""blabla"""


def poly_integral(poly, C=0):
    """
    Args: ff
    Returns: ff
    """
    if not isinstance(poly, list) or not all(
            isinstance(c, (int, float)) for c in poly):
        return None
    integral_poly = []
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        if new_coeff == int(new_coeff):
            integral_poly.append(int(new_coeff))
        else:
            integral_poly.append(new_coeff)
    return integral_poly
