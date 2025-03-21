#!/usr/bin/env python3
"""fonction"""


class Exponential:
    """class de la loi exepenentiel"""
    def __init__(self, data=None, lambtha=1.):
        """initialisation du code"""
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
            self.lambtha = float(1/(sum(data) / len(data)))

    def pdf(self, x):
        """Probabilité de densité de fonction"""
        e = 2.7182818285
        if x < 0:
            return 0
        pdf_value = e**-(self.lambtha*x)
        return pdf_value * self.lambtha

    def cdf(self, x):
        """Cumulative Distribution Function"""
        e = 2.7182818285
        if x < 0:
            return 0
        cdf_values = e**-(self.lambtha*x)
        return 1 - cdf_values
