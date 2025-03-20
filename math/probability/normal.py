#!/usr/bin/env python3
"""fonction"""


class Normal:
    """class de la loi normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("lambtha must be a positive value")
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if not len(data) >= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float((sum(data))/len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
