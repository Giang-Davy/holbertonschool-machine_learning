#!/usr/bin/env python3
"""blabla"""

def poly_integral(poly, C=0):
    """
    Args: 
        poly (list of int or float): List of coefficients representing a polynomial.
        C (int or float, optional): The constant of integration. Defaults to 0.
    Returns:
        list or None: A list of coefficients representing the integral of the polynomial, or None if input is invalid.
    """
    # Check if poly is a list and C is a number (int or float)
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly) or not isinstance(C, (int, float)):
        return None

    integral_poly = [C]  # Initialize the integral with the constant of integration

    # For each coefficient, integrate by dividing it by the new exponent
    for i in range(len(poly)):
        if poly[i] != 0:  # Skip zero coefficients as they do not affect the integral
            new_coeff = poly[i] / (i + 1)  # The new coefficient after integration
            # If the new coefficient is an integer, we store it as an integer
            if new_coeff == int(new_coeff):
                integral_poly.append(int(new_coeff))
            else:
                integral_poly.append(new_coeff)

    return integral_poly
