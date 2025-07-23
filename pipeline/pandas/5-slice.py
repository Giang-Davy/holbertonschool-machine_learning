#!/usr/bin/env python3
"""5-slice.py"""


def slice(df):
    """découpage"""

    valeur_select = df[["High", "Low", "Close", "Volume_(BTC)"]]
    return valeur_select.iloc[::60]
