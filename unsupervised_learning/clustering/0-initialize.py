#!/usr/bin/env python3
"""0-initialize.py"""


import numpy as np
import matplotlib.pyplot as plt


def initialize(X, k):
    """initialisation"""
    mini = np.min(X, axis=0)
    max = np.max(X, axis=0)
    uni = np.ramdom.uniform(mini, max)
    return uni
