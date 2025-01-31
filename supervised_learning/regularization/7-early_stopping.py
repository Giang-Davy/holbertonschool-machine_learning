#!/usr/bin/env python3
"""fonction"""


import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """stopper en avance"""
    if opt_cost - cost > threshold:
        return (False, 0)
    else:
        count += 1
        if count >= patience:
            return (True, count)
        return (False, count)
