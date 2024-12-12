#!/usr/bin/env python3
"""blabla"""


def poly_derivative(poly):
    """
    Args: ff
    Returns: ff
    """
    if not isinstance(poly, list)
    or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    return [i * poly[i] for i in range(1, len(poly))]
