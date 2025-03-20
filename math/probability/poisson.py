#!/usr/bin/env python3
"""fonction"""


class Poisson:
    """class de la loi Poissson"""
    def __init__(self, data=None, lambtha=1.):
        """initialisation"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if not len(data) >= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
