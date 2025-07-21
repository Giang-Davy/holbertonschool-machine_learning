#!/usr/bin/env python3
"""0-from_numpy.py"""


import pandas as pd


def from_numpy(array):
    """création de dataframe"""
    liste = []
    colonnes = array.shape[1]
    for i in range(colonnes):
        lettre = chr(i+65)
        liste.append(lettre)
    data_frame = pd.DataFrame(data=array, columns=liste)

    return data_frame
