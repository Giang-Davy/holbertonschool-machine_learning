#!/usr/bin/env python3


import numpy as np


class Neuron:
    """
    Classe qui présente les neuronnes
    """
    def __init__(self, nx):
        """
        Args: ff
        Returns: ff
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive")
        self._W = np.random.randn(1, nx)
        self._b = 0
        self._A = 0

    @property
    def W(self):
        return self._W
    
    @property
    def b(self):
        return self._b
    
    @property
    def A(self):
        return self._A
