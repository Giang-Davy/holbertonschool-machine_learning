#!/usr/bin/env python3
"""11-concat.py"""


import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ranger"""

    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[df1.index.to_series().between(1417411980, 1417417980)]
    df2 = df2.loc[df2.index.to_series().between(1417411980, 1417417980)]
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.swaplevel(0, 1)
    df = df.sort_index(level=0)

    return df
