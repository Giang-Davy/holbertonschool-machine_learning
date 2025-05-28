#!/usr/bin/env python3
"""4-positional_encoding.py"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """encodement"""
    pos = np.arange(max_seq_len)
    dm_matrice = np.arange(0, dm, 2)
    pos = np.expand_dims(pos, axis=1)
    PE_1 = np.sin(pos/(10000**(dm_matrice/dm)))
    PE_2 = np.cos(pos/(10000**(dm_matrice/dm)))
    result = np.concatenate([PE_1, PE_2], axis=1)
    result[:, 0::2] = PE_1
    result[:, 1::2] = PE_2
    return result
