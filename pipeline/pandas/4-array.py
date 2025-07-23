#!/usr/bin/env python3
"""4-array.py"""


import pandas as pd


def array(df):
    """tableau"""
    valeur_select = df[["High", "Close"]].tail(10)
    return valeur_select.values
