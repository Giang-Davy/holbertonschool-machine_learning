#!/usr/bin/env python3
"""2-from_file.py"""


import pandas as pd


def from_file(filename, delimiter):
    """charger des données"""
    df = pd.read_csv(filepath_or_buffer=filename, delimiter=delimiter)
    return df
