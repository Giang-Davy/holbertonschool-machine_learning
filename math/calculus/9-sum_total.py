#!/usr/bin/env python3
"""ff"""


def summation_i_squared(n):
    """
    Argsfqsfqs
    """
    if not isinstance(n, int):
       return None
    else:
        return sum(i ** 2 for i in range(1, n + 1))
