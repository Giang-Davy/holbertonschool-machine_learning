#!/usr/bin/env python3
"""11-concat.py"""


import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """concatener"""

    df2 = df2.loc[df2["Timestamp"] <= 1417411920]
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return df
