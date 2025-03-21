#!/usr/bin/env python3
"""fonction"""


class Binomial:
    """code pour la loi binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n = int(n)
            self.p = float(p)
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < self.p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - (variance / mean)
            n = mean / p
            n = round(n)
            p = mean / n
            self.n = int(n)
            self.p = float(p)
