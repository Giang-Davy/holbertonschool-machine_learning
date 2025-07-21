#!/usr/bin/env python3
"""0-from_numpy.py"""


import pandas as pd


def from_numpy(array):
    """création de dataframe"""

    data_frame = pd.DataFrame(data=array)

    return data_frame
