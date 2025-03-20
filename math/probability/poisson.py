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

    def pmf(self, k):
        """Probability Mass Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorielle = 1
        for i in range(1, k + 1):
            factorielle *= i
        e = 2.7182818285
        pmf_result = (e**-self.lambtha*self.lambtha**k)/factorielle
        return pmf_result

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
