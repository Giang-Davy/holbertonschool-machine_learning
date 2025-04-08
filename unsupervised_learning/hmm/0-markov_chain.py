#!/usr/bin/env python3


import numpy as np


def markov_chain(P, s, t=1):
    """chaine de markov"""
    if not isinstance(P, np.ndarray):
        return None
    for i in range(t):
        s = s @ P
    return s 
