#!/usr/bin/env python3
"""fonction"""


import numpy as np


def normalize(X, m, s):
    """normal"""
    X_normalize = (X - m)/s
    return X_normalize
