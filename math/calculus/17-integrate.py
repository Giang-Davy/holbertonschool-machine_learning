#!/usr/bin/env python3
"""fonction"""


def poly_integral(poly, C=0):
    """
    Args: ff
    Returns: ff
    """
    if not isinstance(poly, list) or not all(
            isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        if coeff != 0:
            integral.append(coeff / (i + 1))

    return integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
