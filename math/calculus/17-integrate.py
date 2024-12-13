#!/usr/bin/env python3
"""blabla"""


def poly_integral(poly, C=0):
    """
    Args: ff
    Returns: ff
    """
    if not isinstance(poly, list) or not all(
            isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]

    for i in range(len(poly)):
        integral.append(poly[i] / (i + 1))

    while len(integral) < len(poly) + 1:
        integral.append(0)

    integral = [int(x) if isinstance(
        x, float) and x.is_integer() else x for x in integral]

    return integral
