#!/usr/bin/env python3
"""blablabla"""


import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Args: ff

    Returns: ff
    """
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='r', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
